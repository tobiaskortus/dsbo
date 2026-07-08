from __future__ import annotations

import io
import os
from itertools import groupby
from typing import TYPE_CHECKING

import matplotlib
import matplotlib.colors as mcolors
import matplotlib.patches as mpatches
import matplotlib.pyplot as plt
from graphviz import Digraph

if TYPE_CHECKING:
    from IPython.display import Image, display
    from dsbo.workflow import Workflow

# task_levels is imported lazily inside visualize_workflow to avoid circular import


def visualize_workflow(
    workflow: Workflow,
    filename: str | None = None,
    format: str = "png",
):
    """Visualize the task graph of a workflow using graphviz and matplotlib."""

    def _res_key(n):
        res = getattr(n, "resources", None)
        return "default" if res is None else f"mem={res.memory}, cpu={res.job_cpu}, gpu={res.gpu_resources}"

    def _node_label(n):
        name = n.name if hasattr(n, "name") else str(n)
        return f"{name}\n{_res_key(n)}"

    def _build_color_map(nodes):
        keys = list(dict.fromkeys(_res_key(n) for n in nodes))
        cmap = matplotlib.colormaps["Set3"]
        color_map = {k: mcolors.to_hex(cmap(i % cmap.N)) for i, k in enumerate(keys)}
        keys = [k for k in keys if k != "default"] + (["default"] if "default" in keys else [])
        return keys, color_map

    def _build_dep_group_key(nodes):
        lookup = {
            node: gi
            for gi, dg in enumerate(workflow._dependency_groups)
            for node in dg.nodes
            if node in nodes
        }
        return lambda n: ("dg", lookup[n]) if n in lookup else ("single", id(n))

    def _no_grad(n):
        return getattr(n, "_no_grad_op", False)

    def _faded(hex_color, blend=0.7):
        r, g, b = mcolors.to_rgb(hex_color)
        r2 = r + (1 - r) * blend
        g2 = g + (1 - g) * blend
        b2 = b + (1 - b) * blend
        return mcolors.to_hex((r2, g2, b2))

    def _add_node(sg, n):
        nid = str(id(n))
        fill = _faded(_color(n)) if _no_grad(n) else _color(n)
        style = "filled,dashed" if _no_grad(n) else "filled"
        sg.node(
            nid,
            label=_node_label(n),
            style=style,
            fillcolor=fill,
            shape="box",
            fontcolor="black",
            color="black",
        )
        rep_entry[n] = rep_exit[n] = nid

    def _add_replica_group(sg, first, last, count):
        _add_node(sg, first)
        _add_node(sg, last)
        dot.edge(
            str(id(first)),
            str(id(last)),
            label=f"\u00d7{count}",
            style="dashed",
            arrowhead="none",
            constraint="false",
            color="black",
            fontcolor="black",
            penwidth="1",
            fontsize="11",
            labelfloat="false",
        )

    def _render_with_legend(out_format):
        graph_bytes = dot.pipe(format="png")
        if not keys:
            return graph_bytes if out_format == "png" else dot.pipe(format=out_format)

        graph_img = plt.imread(io.BytesIO(graph_bytes))
        h, w = graph_img.shape[:2]
        dpi, legend_h = 100, 0.55
        fig_w, fig_h = w / dpi, h / dpi + legend_h

        fig = plt.figure(figsize=(fig_w, fig_h), dpi=dpi)
        ax_g = fig.add_axes([0, legend_h / fig_h, 1, h / dpi / fig_h])
        ax_g.imshow(graph_img)
        ax_g.axis("off")

        ax_l = fig.add_axes([0, 0, 1, legend_h / fig_h])
        ax_l.axis("off")
        worker_patches = [
            mpatches.Patch(
                facecolor=color_map[k],
                edgecolor="#aaaaaa",
                linewidth=0.5,
                label=k,
            )
            for k in keys
        ]

        has_no_grad = any(_no_grad(n) for n in nodes)
        has_grad = any(not _no_grad(n) for n in nodes)

        worker_legend = ax_l.legend(
            handles=worker_patches,
            ncol=min(4, len(worker_patches)),
            loc="center",
            frameon=False,
            fontsize=8,
            handlelength=1.2,
            handleheight=0.9,
            columnspacing=1.2,
            handletextpad=0.5,
            title="Compute Resources",
            title_fontsize=8,
        )
        worker_legend.get_title().set_fontweight("bold")

        if has_no_grad and has_grad:
            grad_patches = [
                mpatches.Patch(
                    facecolor="white",
                    edgecolor="black",
                    linewidth=1,
                    linestyle="solid",
                    label="grad",
                ),
                mpatches.Patch(
                    facecolor="white",
                    edgecolor="black",
                    linewidth=1,
                    linestyle="dashed",
                    label="no grad",
                ),
            ]
            ax_l.add_artist(worker_legend)
            ax_l.legend(
                handles=grad_patches,
                ncol=1,
                loc="center right",
                frameon=False,
                fontsize=8,
                handlelength=1.2,
                handleheight=0.9,
                columnspacing=1.2,
                handletextpad=0.5,
            )

        buf = io.BytesIO()
        fig.savefig(buf, format=out_format, bbox_inches="tight", dpi=dpi)
        plt.close(fig)
        buf.seek(0)
        return buf.read()

    fmt = format
    if filename:
        base, ext = os.path.splitext(filename)
        if ext:
            filename, fmt = base, ext.lstrip(".")

    dot = Digraph(format=fmt)
    dot.attr(rankdir="LR")
    dot.attr("node", fontsize="10")
    dot.attr("edge", arrowsize="0.6")

    nodes = set(workflow.task_graph.nodes())
    keys, color_map = _build_color_map(nodes)
    _color = lambda n: color_map.get(_res_key(n), "#d3d3d3")
    _dep_group_key = _build_dep_group_key(nodes)
    from dsbo.workflow import task_levels
    levels = task_levels(workflow.task_graph)

    rep_entry: dict = {}
    rep_exit: dict = {}
    hidden_nodes: set = set()

    level_groups: dict[int, list] = {}
    for n in nodes:
        level_groups.setdefault(levels.get(n, 0), []).append(n)

    for lvl, group in sorted(level_groups.items()):
        group.sort(
            key=lambda n: (
                getattr(n, "i", None)
                if isinstance(getattr(n, "i", None), int)
                else float("inf"),
                n.name if hasattr(n, "name") else "",
            )
        )
        with dot.subgraph(name=f"rank_{lvl}") as sg:
            sg.attr(rank="same")
            for (kind, _), pg in groupby(group, key=_dep_group_key):
                pg = list(pg)
                if kind == "dg" and len(pg) > 2:
                    hidden_nodes.update(pg[1:-1])
                    _add_replica_group(sg, pg[0], pg[-1], len(pg))
                else:
                    for n in pg:
                        _add_node(sg, n)

    for u, v in workflow.task_graph.edges():
        if u in hidden_nodes or v in hidden_nodes:
            continue
        dot.edge(rep_exit.get(u, str(id(u))), rep_entry.get(v, str(id(v))))

    if filename is not None:
        out_path = f"{filename}.{fmt}"
        with open(out_path, "wb") as f:
            f.write(_render_with_legend(fmt))
        return out_path

    from IPython.display import Image, display

    display(Image(_render_with_legend(format)))
