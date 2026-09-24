from __future__ import annotations

import io
import os
import xml.etree.ElementTree as ET
from typing import TYPE_CHECKING

import networkx as nx
import matplotlib
import matplotlib.colors as mcolors
import matplotlib.patches as mpatches
import matplotlib.pyplot as plt
from graphviz import Digraph

if TYPE_CHECKING:
    from IPython.display import Image, display
    from dsbo.workflow import Workflow

def visualize_workflow(
    workflow: Workflow,
    filename: str | None = None,
    format: str = "png",
):
    """Visualize the task graph of a workflow using graphviz and matplotlib."""

    def _res_key(n):
        res = getattr(n, "resources", None)
        return "default" if res is None else f"mem={res.memory}, cpu={res.cores}"

    def _node_label(n, template=None):
        label_source = template if template is not None else n
        name = label_source.name if hasattr(label_source, "name") else str(label_source)
        is_fused = getattr(n, "_is_batch_fused", False)
        suffix = ""
        if is_fused:
            suffix = " [batch: reduce]"
        return f"{name}{suffix}\n{_res_key(n)}"

    def _build_color_map(nodes):
        keys = list(dict.fromkeys(_res_key(n) for n in nodes))
        cmap = matplotlib.colormaps["Set3"]
        color_map = {k: mcolors.to_hex(cmap(i % cmap.N)) for i, k in enumerate(keys)}
        keys = [k for k in keys if k != "default"] + (["default"] if "default" in keys else [])
        return keys, color_map

    def _no_grad(n):
        return getattr(n, "_no_grad_op", False)

    def _faded(hex_color, blend=0.7):
        r, g, b = mcolors.to_rgb(hex_color)
        r2 = r + (1 - r) * blend
        g2 = g + (1 - g) * blend
        b2 = b + (1 - b) * blend
        return mcolors.to_hex((r2, g2, b2))

    def _add_node(sg, n, template=None):
        nid = str(id(n))
        fill = _faded(_color(n)) if _no_grad(n) else _color(n)
        style = "filled,dashed" if _no_grad(n) else "filled"
        label_text = _node_label(n, template)
        in_chain = n in stacked_nodes
        sg.node(
            nid,
            label=label_text,
            style=style,
            fillcolor=fill,
            shape="box",
            fontcolor="black",
            color="black",
            tooltip="batch_chain" if in_chain else "",
        )
        rep_entry[n] = rep_exit[n] = nid

    def _add_replica_group(sg, first, last, count, template):
        _add_node(sg, first, template)
        _add_node(sg, last, template)
        count_label = (
            '<<TABLE BORDER="0" CELLBORDER="0" CELLSPACING="0" CELLPADDING="0">'
            f'<TR><TD WIDTH="28" ALIGN="RIGHT">\u00d7{count}</TD>'
            '<TD WIDTH="8">&#160;</TD></TR></TABLE>>'
        )
        sg.edge(
            str(id(first)),
            str(id(last)),
            label=count_label,
            style="dashed",
            arrowhead="none",
            constraint="false",
            color="black",
            fontcolor="black",
            penwidth="1",
            fontsize="11",
            labelfloat="false",
        )

    def _stack_svg_nodes(svg_text):
        """Clone batch-chain node <g> elements to create a stacked-card look.

        Parses SVG with xml.etree, clones <g class="node"> elements that
        contain ``xlink:title="batch_chain"`` (non-fused predecessors of
        batch-fused nodes), and inserts copies offset right-and-up behind the
        original node so the stack reads as a pile of cards.
        """
        ns = {"svg": "http://www.w3.org/2000/svg", "xlink": "http://www.w3.org/1999/xlink"}
        ET.register_namespace("", ns["svg"])
        ET.register_namespace("xlink", ns["xlink"])
        root = ET.fromstring(svg_text)
        graph0 = root.find(".//svg:g[@id='graph0']", ns)
        if graph0 is None:
            return svg_text

        to_clone: list[ET.Element] = []
        for g_node in graph0.findall("svg:g", ns):
            if g_node.get("class") != "node":
                continue
            a = g_node.find(".//svg:a", ns)
            if a is not None and a.get(f"{{{ns['xlink']}}}title") == "batch_chain":
                to_clone.append(g_node)

        if not to_clone:
            return svg_text

        import copy
        for node_g in to_clone:
            for dx in (4, 2):
                clone = copy.deepcopy(node_g)
                dy = -dx
                t = clone.get("transform", "")
                clone.set("transform", f"translate({dx},{dy}){' ' + t if t else ''}")
                graph0.insert(list(graph0).index(node_g), clone)

        return ET.tostring(root, encoding="unicode")

    def _svg_to_png(svg_text):
        try:
            import cairosvg
            return cairosvg.svg2png(bytestring=svg_text.encode("utf-8"))
        except ImportError:
            tmp = f"/tmp/_viz_{os.getpid()}.svg"
            with open(tmp, "w") as f:
                f.write(svg_text)
            import subprocess
            subprocess.run(["rsvg-convert", tmp, "-o", tmp.replace(".svg", ".png")],
                           capture_output=True)
            with open(tmp.replace(".svg", ".png"), "rb") as f:
                return f.read()

    def _render_with_legend(out_format):
        svg_bytes = dot.pipe(format="svg")
        svg_text = svg_bytes.decode("utf-8") if isinstance(svg_bytes, bytes) else svg_bytes
        svg_text = _stack_svg_nodes(svg_text)
        png_bytes = _svg_to_png(svg_text)

        if not keys:
            if out_format == "svg":
                return svg_text.encode("utf-8")
            return png_bytes

        graph_img = plt.imread(io.BytesIO(png_bytes))
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
    dot.attr(rankdir="LR", newrank="true")
    dot.attr("node", fontsize="10")
    dot.attr("edge", arrowsize="0.6")

    nodes = set(workflow.task_graph.nodes())
    stacked_nodes = {
        ancestor
        for fused in nodes
        if getattr(fused, "_is_batch_fused", False)
        for ancestor in nx.ancestors(workflow.task_graph, fused)
        if not getattr(ancestor, "_is_batch_fused", False)
    }
    keys, color_map = _build_color_map(nodes)
    _color = lambda n: color_map.get(_res_key(n), "#d3d3d3")
    from dsbo.workflow import task_levels
    levels = task_levels(workflow.task_graph)

    from dsbo.surrogates.graph_transformations import GraphMode
    is_default_mode = workflow.graph_context.mode == GraphMode.DEFAULT

    dependency_groups = workflow.surrogate_execution.dependency_groups if is_default_mode \
        else getattr(workflow.task_graph, "_dependency_groups", ())

    rep_entry: dict = {}
    rep_exit: dict = {}
    hidden_nodes: set = set()
    grouped_nodes: set = set()

    if is_default_mode or dependency_groups:
        for gi, dependency_group in enumerate(dependency_groups):
            active_nodes = dependency_group.nodes & nodes
            if not active_nodes:
                continue

            grouped_nodes.update(active_nodes)
            template_replicas: dict[object, list] = {}
            for node in active_nodes:
                template = dependency_group.node_to_template.get(node, node)
                template_replicas.setdefault(template, []).append(node)

            with dot.subgraph(name=f"cluster_dependency_{gi}") as sg:
                sg.attr(
                    color="#777777",
                    style="dashed",
                )
                for template, replicas in template_replicas.items():
                    replicas.sort(key=lambda n: getattr(n, "i", 0) or 0)
                    if len(replicas) > 1:
                        hidden_nodes.update(replicas[1:-1])
                        _add_replica_group(
                            sg,
                            replicas[0],
                            replicas[-1],
                            dependency_group.count,
                            template,
                        )
                    else:
                        for node in replicas:
                            _add_node(sg, node, template)

    for node in nodes - grouped_nodes:
        _add_node(dot, node)

    level_groups: dict[int, list] = {}
    for node in nodes - hidden_nodes:
        level_groups.setdefault(levels.get(node, 0), []).append(node)

    for level, group in sorted(level_groups.items()):
        with dot.subgraph(name=f"rank_{level}") as sg:
            sg.attr(rank="same")
            for node in group:
                sg.node(str(id(node)))

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
