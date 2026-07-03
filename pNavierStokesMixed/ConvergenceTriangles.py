from fractions import Fraction

import numpy as np
from matplotlib.font_manager import FontProperties
from matplotlib.offsetbox import AnnotationBbox, DrawingArea
from matplotlib.patches import FancyBboxPatch, PathPatch, Polygon
from matplotlib.textpath import TextPath
from matplotlib.transforms import Affine2D


LABEL_BACKGROUND = '#f0f0f0'
LABEL_EDGE = '0.55'
LABEL_ALPHA = 0.94


def style_legend(legend):
    if legend is None:
        return None
    frame = legend.get_frame()
    frame.set_facecolor(LABEL_BACKGROUND)
    frame.set_edgecolor(LABEL_EDGE)
    frame.set_alpha(LABEL_ALPHA)
    frame.set_linewidth(0.7)
    legend.set_zorder(30)
    return legend


def _add_boxed_label(ax, label, xy, fontsize, xybox=(0.0, 0.0), box_alignment=(0.5, 0.5)):
    scale = ax.figure.dpi / 72.0
    text_path = TextPath((0.0, 0.0), label, size=fontsize, prop=FontProperties(family='serif'))
    text_bbox = text_path.get_extents()
    pad_x = 2.3 * scale
    pad_y = 1.6 * scale
    width = text_bbox.width * scale + 2.0 * pad_x
    height = text_bbox.height * scale + 2.0 * pad_y

    drawing_area = DrawingArea(width, height, 0.0, 0.0)
    drawing_area.add_artist(FancyBboxPatch(
        (0.0, 0.0),
        width,
        height,
        boxstyle=f'round,pad=0,rounding_size={1.8 * scale}',
        facecolor=LABEL_BACKGROUND,
        edgecolor=LABEL_EDGE,
        linewidth=0.55 * scale,
        alpha=LABEL_ALPHA,
    ))
    centered_text_path = text_path.transformed(
        Affine2D()
        .scale(scale)
        .translate(
            0.5 * (width - text_bbox.width * scale) - text_bbox.x0 * scale,
            0.5 * (height - text_bbox.height * scale) - text_bbox.y0 * scale,
        )
    )
    drawing_area.add_artist(PathPatch(
        centered_text_path,
        facecolor='black',
        edgecolor='none',
    ))

    annotation = AnnotationBbox(
        drawing_area,
        xy,
        xybox=xybox,
        xycoords='data',
        boxcoords='offset points',
        box_alignment=box_alignment,
        frameon=False,
        pad=0.0,
        annotation_clip=False,
        zorder=3.0,
    )
    ax.add_artist(annotation)
    return annotation


def _rate_label(rate):
    fraction = Fraction(float(rate)).limit_denominator(8)
    value = fraction.numerator / fraction.denominator
    if abs(value - rate) > 1e-10:
        return rf'${rate:.2g}$'
    if fraction.denominator == 1:
        return str(fraction.numerator)
    return rf'$\frac{{{fraction.numerator}}}{{{fraction.denominator}}}$'


def _valid_xy(x_values, y_values):
    x = np.asarray(x_values, dtype=float)
    y = np.asarray(y_values, dtype=float)
    mask = np.isfinite(x) & np.isfinite(y) & (x > 0.0) & (y > 0.0)
    x = x[mask]
    y = y[mask]
    order = np.argsort(x)
    return x[order], y[order]


def _estimate_rate(x_values, y_values, tail=1):
    x, y = _valid_xy(x_values, y_values)
    if len(x) < 2:
        return None

    stop = min(len(x), tail + 1)
    x_tail = x[:stop]
    y_tail = y[:stop]
    slopes = np.diff(np.log(y_tail)) / np.diff(np.log(x_tail))
    slopes = slopes[np.isfinite(slopes) & (slopes > 0.0)]
    if len(slopes) == 0:
        return None
    return float(np.median(slopes))


def _rounded_rate(rate, step=0.25):
    if rate is None or not np.isfinite(rate) or rate <= 0.0:
        return None
    return max(step, round(rate / step) * step)


def _geometric_mean(values):
    values = np.asarray(values, dtype=float)
    values = values[np.isfinite(values) & (values > 0.0)]
    if len(values) == 0:
        return None
    return float(np.exp(np.mean(np.log(values))))


def _log_intervals_overlap(first, second):
    first_low, first_high = first
    second_low, second_high = second
    return max(np.log(first_low), np.log(second_low)) < min(np.log(first_high), np.log(second_high))


def _log_rectangles_overlap(first, second):
    return (_log_intervals_overlap(first['x_interval'], second['x_interval'])
            and _log_intervals_overlap(first['y_interval'], second['y_interval']))


def _curve_lower_edge_requirement(curve, x_left, x_right, rate, vertical_gap):
    mask = (curve['x'] >= x_left) & (curve['x'] <= x_right)
    if not np.any(mask):
        return curve['fine_value'] * vertical_gap

    x_span = curve['x'][mask]
    y_span = curve['y'][mask]
    scale = (x_span / x_left) ** rate
    return float(np.max(vertical_gap * y_span / scale))


def _curve_sloped_edge_below_requirement(curve, x_left, x_right, rate, vertical_gap):
    mask = (curve['x'] >= x_left) & (curve['x'] <= x_right)
    if not np.any(mask):
        return curve['fine_value'] / vertical_gap

    x_span = curve['x'][mask]
    y_span = curve['y'][mask]
    scale = (x_span / x_left) ** rate
    return float(np.min(y_span / (vertical_gap * scale)))


def _triangle_extent(positive_x, rate, max_span, max_height_factor, start_index=0):
    start_index = max(0, min(start_index, len(positive_x) - 2))
    x_left = positive_x[start_index]
    last_index = min(start_index + max_span, len(positive_x) - 1)
    if max_height_factor is None:
        x_right = positive_x[last_index]
        return x_left, x_right, x_right / x_left

    for index in range(last_index, start_index, -1):
        x_right = positive_x[index]
        q = x_right / x_left
        if q ** rate <= max_height_factor:
            return x_left, x_right, q

    x_right = positive_x[start_index + 1]
    return x_left, x_right, x_right / x_left


def draw_convergence_triangles(
        ax,
        x_values,
        curves,
        tail=1,
        span=3,
        rate_step=0.25,
        vertical_gap=1.55,
        stack_factor=1.65,
        max_height_factor=None,
        face_alpha=0.92,
        fontsize=20,
        label_fontsize=10.5):
    x = np.asarray(x_values, dtype=float)
    positive_x = np.sort(np.unique(x[np.isfinite(x) & (x > 0.0)]))
    if len(positive_x) < 2:
        return []

    groups = {}
    valid_curves = []
    for curve in curves:
        y_raw = np.asarray(curve['values'], dtype=float)
        x_curve, y_curve = _valid_xy(x, y_raw)
        positive_y = y_curve[np.isfinite(y_curve) & (y_curve > 0.0)]
        if len(positive_y) == 0:
            continue

        estimated_rate = _estimate_rate(x, y_raw, tail=tail)
        reference_rate = curve.get('rate', estimated_rate)
        rate = _rounded_rate(reference_rate, step=rate_step)
        curve_data = {
            'color': curve['color'],
            'x': x_curve,
            'y': y_curve,
            'fine_value': float(y_curve[0]),
            'estimated_rate': estimated_rate,
            'side': curve.get('side', 'above'),
            'start_index': int(curve.get('start_index', 0)),
            'vertical_gap': float(curve.get('vertical_gap', vertical_gap)),
        }
        valid_curves.append(curve_data)
        if rate is None:
            continue
        groups.setdefault(rate, []).append(curve_data)

    if not groups or not valid_curves:
        return []

    occupied = []
    drawn = []
    hatch_patterns = [r'\\', '//', 'xx']

    sorted_groups = sorted(
        groups.items(),
        key=lambda item: _geometric_mean([curve['fine_value'] for curve in item[1]]),
    )

    for rate, group in sorted_groups:
        y_ref = _geometric_mean([curve['fine_value'] for curve in group])
        if y_ref is None:
            continue

        side = group[0]['side']
        group_gap = group[0]['vertical_gap']
        start_index = group[0]['start_index']
        x_left, x_right, q = _triangle_extent(
            positive_x,
            rate,
            span,
            max_height_factor,
            start_index=start_index,
        )
        if side == 'below':
            y_low = min(
                _curve_sloped_edge_below_requirement(curve, x_left, x_right, rate, group_gap)
                for curve in group
            )
            y_base = y_low * (q ** rate)
        else:
            y_low = max(_curve_lower_edge_requirement(curve, x_left, x_right, rate, group_gap)
                        for curve in group)
            y_low = max(y_low, y_ref * group_gap)
            y_base = y_low * (q ** rate)
            y_low = y_base / (q ** rate)
        candidate = {
            'x_interval': (x_left, x_right),
            'y_interval': (y_low, y_base),
        }
        while any(_log_rectangles_overlap(candidate, interval) for interval in occupied):
            blocking_intervals = [
                interval
                for interval in occupied
                if _log_intervals_overlap(candidate['x_interval'], interval['x_interval'])
            ]
            if side == 'below':
                blocking_low = min(interval['y_interval'][0] for interval in blocking_intervals)
                y_base = blocking_low / stack_factor
                y_low = y_base / (q ** rate)
            else:
                blocking_high = max(interval['y_interval'][1] for interval in blocking_intervals)
                y_low = blocking_high * stack_factor
                y_base = y_low * (q ** rate)
            candidate['y_interval'] = (y_low, y_base)

        if side == 'below':
            points = np.array([
                [x_left, y_low],
                [x_right, y_base],
                [x_right, y_low],
            ])
            rate_label_xy = (x_right, np.sqrt(y_low * y_base))
            unit_label_xy = (np.sqrt(x_left * x_right), y_low)
        else:
            points = np.array([
                [x_left, y_base],
                [x_right, y_base],
                [x_left, y_low],
            ])
            rate_label_xy = (x_left, np.sqrt(y_low * y_base))
            unit_label_xy = (np.sqrt(x_left * x_right), y_base)

        colors = [curve['color'] for curve in group]
        ax.add_patch(Polygon(
            points,
            closed=True,
            facecolor=colors[0],
            edgecolor='none',
            alpha=face_alpha,
            zorder=1.4,
        ))

        for index, color in enumerate(colors[1:]):
            ax.add_patch(Polygon(
                points,
                closed=True,
                facecolor='none',
                edgecolor=color,
                linewidth=0.01,
                hatch=hatch_patterns[index % len(hatch_patterns)],
                zorder=1.5,
            ))

        ax.add_patch(Polygon(
            points,
            closed=True,
            facecolor='none',
            edgecolor='k',
            linewidth=1.0,
            zorder=1.6,
        ))
        ax.update_datalim(points)
        ax.autoscale_view()

        _add_boxed_label(
            ax,
            _rate_label(rate),
            rate_label_xy,
            label_fontsize,
        )
        _add_boxed_label(
            ax,
            '1',
            unit_label_xy,
            label_fontsize,
        )

        occupied.append(candidate)
        drawn.append({
            'rate': rate,
            'estimated_rates': [curve['estimated_rate'] for curve in group],
            'colors': colors,
            'x_left': x_left,
            'x_right': x_right,
            'y_low': y_low,
            'y_base': y_base,
        })

    return drawn
