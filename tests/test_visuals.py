"""Tests for koyo.visuals."""

import matplotlib.pyplot as plt
import numpy as np
import pytest
from koyo.visuals import (
    _get_alternate_locations,
    _plot_image,
    _plot_line,
    _plot_or_update_image,
    add_ax_colorbar,
    add_contour_labels,
    add_contours,
    add_patches,
    add_scalebar,
    auto_clear_axes,
    clear_axes,
    close_mpl_figure,
    compute_divider,
    convert_divider_to_str,
    convert_to_vertical_line_input,
    despine,
    disable_bomb_protection,
    fig_to_pil,
    fix_style,
    get_intensity_formatter,
    get_row_col,
    get_ticks_with_unit,
    inset_colorbar,
    is_dark,
    make_legend_handles,
    pil_to_fig,
    plot_centroids,
    save_gray,
    save_rgb,
    set_intensity_formatter,
    set_tick_fmt,
    shorten_style,
    tight_layout,
    update_colorbar,
    vertices_to_collection,
    y_tick_fmt,
)
from PIL import Image


def test_tight_layout_and_close_mpl_figure():
    fig, ax = plt.subplots()
    ax.plot([0, 1], [0, 1])
    tight_layout(fig)
    number = fig.number
    close_mpl_figure(fig)
    assert not plt.fignum_exists(number)


def test_disable_bomb_protection_restores_setting():
    original = Image.MAX_IMAGE_PIXELS
    with disable_bomb_protection():
        assert Image.MAX_IMAGE_PIXELS == 30_000 * 30_000
    assert original == Image.MAX_IMAGE_PIXELS


def test_save_rgb_and_gray(tmp_path):
    rgb_path = tmp_path / "rgb.png"
    gray_path = tmp_path / "gray.png"
    save_rgb(rgb_path, np.zeros((5, 5, 3), dtype=np.uint8))
    save_gray(gray_path, np.ones((5, 5), dtype=float))
    assert rgb_path.exists()
    assert gray_path.exists()


def test_set_tick_fmt_and_despine():
    fig, ax = plt.subplots()
    returned = set_tick_fmt(ax, use_offset=False, axis="both")
    despine(ax, "horizontal")
    assert returned is ax
    assert not ax.spines["left"].get_visible()
    close_mpl_figure(fig)


def test_fig_to_pil_and_pil_to_fig():
    fig, ax = plt.subplots()
    ax.plot([0, 1], [0, 1])
    image = fig_to_pil(fig)
    assert image.size[0] > 0
    fig2 = pil_to_fig(image)
    assert len(fig2.axes) == 1
    close_mpl_figure(fig2)


def test_clear_axes_and_auto_clear_axes():
    fig, axs = plt.subplots(1, 3)
    axs[0].plot([0, 1], [0, 1])
    clear_axes(0, axs)
    assert not axs[1].axison
    close_mpl_figure(fig)

    fig2, axs2 = plt.subplots(1, 2)
    axs2[0].scatter([0], [0])
    auto_clear_axes(axs2)
    assert axs2[0].axison
    assert not axs2[1].axison
    close_mpl_figure(fig2)


def test_convert_to_vertical_line_input_and_collection():
    x = np.array([1.0, 2.0])
    y = np.array([3.0, 4.0])
    vertices = convert_to_vertical_line_input(x, y)
    assert len(vertices) == 2
    collection = vertices_to_collection(x, y, color="red", line_width=2)
    assert len(collection.get_segments()) == 2


def test_plot_centroids_creates_collection():
    x = np.array([0.0, 1.0, 2.0])
    y = np.array([1.0, 3.0, 2.0])
    fig, ax = plot_centroids(x, y, x_label="x", y_label="y", title="title")
    assert len(ax.collections) == 1
    assert ax.get_title() == "title"
    close_mpl_figure(fig)


@pytest.mark.parametrize(
    ("value", "expected"),
    [
        (0, 1),
        (1500, 3),
        (2_000_000, 6),
    ],
)
def test_compute_divider(value, expected):
    assert compute_divider(value) == expected


@pytest.mark.parametrize(
    ("value", "exp_value", "expected"),
    [
        (0.005, 0, "0.005"),
        (1500, 3, "1.5k"),
        (2_000_000, 6, "2.0M"),
        (3_000_000_000, 9, "3.0B"),
    ],
)
def test_convert_divider_to_str(value, exp_value, expected):
    assert convert_divider_to_str(value, exp_value) == expected


def test_y_tick_fmt_and_intensity_formatter():
    assert y_tick_fmt(1500) == "1.5k"
    formatter = get_intensity_formatter()
    assert formatter(1500, 0) == "1.5k"


def test_set_intensity_formatter():
    fig, ax = plt.subplots()
    set_intensity_formatter(ax)
    assert ax.yaxis.get_major_formatter()(1500, 0) == "1.5k"
    close_mpl_figure(fig)


def test_add_contours_and_alternate_locations():
    contour_a = np.array([[0, 0], [1, 0], [1, 1]])
    contour_b = np.array([[0, 2], [1, 2], [1, 3]])
    locations = _get_alternate_locations({"b": contour_b, "a": contour_a})
    assert locations == {"a": "top", "b": "bottom"}

    fig, ax = plt.subplots()
    add_contours(ax, {"a": contour_a, "b": contour_b})
    add_contour_labels(ax, {"a": contour_a, "b": contour_b}, {"a": "A", "b": "B"})
    assert len(ax.lines) == 2
    assert len(ax.texts) == 2
    close_mpl_figure(fig)


def test_add_ax_colorbar_and_inset_colorbar():
    fig, ax = plt.subplots()
    image = ax.imshow(np.arange(4).reshape(2, 2))
    cbar = add_ax_colorbar(image)
    assert cbar is not None

    ax2, cax, inset = inset_colorbar(ax, image, ticks=[0, 3], ticklabels=["low", "high"], label="Intensity")
    assert ax2 is ax
    assert inset.ax is cax
    assert inset.ax.get_xlabel() == "Intensity" or inset.ax.get_ylabel() == "Intensity"
    close_mpl_figure(fig)


def test_make_legend_handles_and_invalid_kind():
    handles = make_legend_handles(["a", "b"], ["red", "blue"], kind=["line", "patch"], width=[1, 2])
    assert len(handles) == 2
    with pytest.raises(ValueError):
        make_legend_handles(["a"], ["red"], kind=["unknown"])


def test_add_patches():
    fig, axs = plt.subplots(1, 2)
    for ax in axs:
        ax.set_ylim(0, 10)
    add_patches(list(axs), [(1, 2), (3, 5)], colors=["red", "blue"])
    assert len(axs[0].patches) == 1
    assert len(axs[1].patches) == 1
    close_mpl_figure(fig)


def test_add_scalebar_without_dependency(monkeypatch, capsys):
    original_import = __import__

    def fake_import(name, *args, **kwargs):
        if name == "matplotlib_scalebar.scalebar":
            raise ImportError("missing")
        return original_import(name, *args, **kwargs)

    monkeypatch.setattr("builtins.__import__", fake_import)
    fig, ax = plt.subplots()
    assert add_scalebar(ax, 1.0) is None
    assert "matplotlib-scalebar not installed" in capsys.readouterr().out
    close_mpl_figure(fig)


def test_get_row_col_and_ticks_with_unit():
    assert get_row_col(5, 2) == (2, 3)
    ticks, labels = get_ticks_with_unit(0, 10, unit=None, n=3)
    assert ticks == [0.0, 5.0, 10.0]
    assert labels == ["0", "5", "10"]

    ticks_unit, labels_unit = get_ticks_with_unit(0, 10, unit="ms", n=4)
    assert len(ticks_unit) == 5
    assert "ms" in ticks_unit
    assert len(labels_unit) == 5


def test_update_colorbar():
    fig, ax = plt.subplots()
    image = ax.imshow(np.arange(4).reshape(2, 2))
    cbar = fig.colorbar(image, ax=ax)
    update_colorbar(cbar, max_val=5, min_val=1, ticks=[1, 5], ticklabels=["low", "high"])
    assert tuple(cbar.mappable.get_clim()) == (1.0, 5.0)
    close_mpl_figure(fig)


def test_fix_style_and_shorten_style():
    fixed_dark = fix_style("dark")
    assert fixed_dark == "dark_background"

    from matplotlib.style import available

    if "seaborn-v0_8-ticks" in available:
        assert fix_style("seaborn-ticks") == "seaborn-v0_8-ticks"
    assert shorten_style("seaborn-v0_8-darkgrid") == "s-darkgrid"


def test_is_dark():
    with plt.style.context("dark_background"):
        assert is_dark()


def test_plot_image_and_update_image():
    array = np.array([[0.0, 1.0], [2.0, 3.0]])
    fig, ax = _plot_image(array, title="Image", colorbar=True, as_title=True)
    assert ax.get_title() == "Image"
    assert len(ax.images) == 1
    assert ax.child_axes

    image, cbar = _plot_or_update_image(ax, array, title="Updated", colorbar=True)
    image2, cbar2 = _plot_or_update_image(ax, array + 1, img=image, cbar=cbar, title="Updated Again")
    assert image2 is image
    assert cbar2 is cbar
    close_mpl_figure(fig)


def test_plot_line():
    fig, ax = _plot_line(np.array([0, 1, 2]), np.array([1, 2, 3]), marker=1, title="Line", x_label="x", y_label="y")
    assert ax.get_title() == "Line"
    assert ax.get_xlabel() == "x"
    assert len(ax.lines) == 2
    close_mpl_figure(fig)


def test_plot_correlation_if_dependencies_present():
    pd = pytest.importorskip("pandas")
    pytest.importorskip("seaborn")
    from koyo.visuals import plot_correlation

    corr = pd.DataFrame([[1.0, 0.5], [0.5, 1.0]], columns=["a", "b"], index=["a", "b"])
    fig = plot_correlation(corr, annot=True)
    assert fig is not None
    close_mpl_figure(fig)
