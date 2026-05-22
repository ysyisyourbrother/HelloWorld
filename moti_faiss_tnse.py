# -*- coding: utf-8 -*-
"""对比多份 Faiss 向量库在二维嵌入中的分布：vrag、symphony_v5，以及可选的单视频全量编码索引。

默认：先对向量做 L2 归一化（与 FlatIP 检索几何一致），再 PCA 预降维后 t-SNE。
依赖：faiss、numpy、matplotlib；降维需 sklearn（Jetson 等设备若未装 sklearn 需自行安装）。
"""
from __future__ import print_function

import argparse
import os
import sys

import faiss
import numpy as np


def parse_args():
    repo = os.path.dirname(os.path.abspath(__file__))
    default_vrag = os.path.join(
        repo, "database", "videomme", "vrag", "short", "faiss", "-XpJeDGh8No.faiss"
    ) # -XpJeDGh8No 这个效果很好
    default_sym = os.path.join(
        repo, "database", "videomme", "symphony_v5", "medium", "faiss", "-XpJeDGh8No.faiss"
    )
    default_out = os.path.join(repo, "moti_faiss_tsne_compare.png")
    p = argparse.ArgumentParser(
        description="将 vrag、symphony 与可选的全量视频 Faiss 索引中的向量映射到二维并绘制对比图"
    )
    p.add_argument("--vrag-faiss", default=default_vrag, help="vrag 侧 Faiss 路径")
    p.add_argument("--symphony-faiss", default=default_sym, help="symphony_v5 侧 Faiss 路径")
    p.add_argument(
        "--full-faiss",
        default=None,
        metavar="PATH",
        help="单视频全量编码的 Faiss 路径；若给出则与上述两份索引合并降维并绘制",
    )
    p.add_argument("-o", "--output", default=default_out, help="输出图像路径")
    p.add_argument(
        "--method",
        choices=("tsne", "pca"),
        default="tsne",
        help="二维映射方法：tsne（默认，先 PCA 再 t-SNE）或仅 pca",
    )
    p.add_argument(
        "--no-normalize",
        action="store_true",
        help="不做逐向量 L2 归一化（默认会做，以贴近内积检索）",
    )
    p.add_argument(
        "--pca-components",
        type=int,
        default=50,
        help="t-SNE 前 PCA 目标维数（仅 method=tsne 时生效）",
    )
    p.add_argument("--random-state", type=int, default=0, help="随机种子")
    p.add_argument(
        "--perplexity",
        type=float,
        default=-1.0,
        help="t-SNE perplexity，<=0 时自动取 min(30, n_samples/8)",
    )
    p.add_argument("--dpi", type=int, default=120, help="输出图像 DPI")
    return p.parse_args()


def _load_vectors_faiss(path):
    if not os.path.isfile(path):
        print("文件不存在: %s" % path, file=sys.stderr)
        sys.exit(1)
    idx = faiss.read_index(path)
    n = int(idx.ntotal)
    d = int(idx.d)
    if n == 0:
        print("索引为空: %s" % path, file=sys.stderr)
        sys.exit(1)
    x = idx.reconstruct_n(0, n)
    x = np.asarray(x, dtype=np.float64)
    if x.ndim == 1:
        x = x.reshape(n, d)
    return x


def _row_l2_normalize(mat):
    norms = np.linalg.norm(mat, axis=1, keepdims=True)
    norms = np.maximum(norms, 1e-12)
    return mat / norms


def _axis_limits_from_xy(xy, pad_frac=0.05):
    xlo = float(np.min(xy[:, 0]))
    xhi = float(np.max(xy[:, 0]))
    ylo = float(np.min(xy[:, 1]))
    yhi = float(np.max(xy[:, 1]))
    rx = max(xhi - xlo, 1e-9)
    ry = max(yhi - ylo, 1e-9)
    pad_x = pad_frac * rx
    pad_y = pad_frac * ry
    return (xlo - pad_x, xhi + pad_x), (ylo - pad_y, yhi + pad_y)


def _matplotlib_cjk_font():
    import matplotlib.font_manager as fm

    prefer = (
        "WenQuanYi Micro Hei",
        "Noto Sans CJK SC",
        "Source Han Sans SC",
        "SimHei",
        "Microsoft YaHei",
    )
    installed = {f.name for f in fm.fontManager.ttflist}
    for name in prefer:
        if name in installed:
            return name
    return None


if __name__ == "__main__":
    args = parse_args()
    xv = _load_vectors_faiss(args.vrag_faiss)
    xs = _load_vectors_faiss(args.symphony_faiss)
    xf = None
    if args.full_faiss is not None:
        xf = _load_vectors_faiss(args.full_faiss)
    dims = [xv.shape[1], xs.shape[1]]
    if xf is not None:
        dims.append(xf.shape[1])
    d_max = max(dims)
    d_min = min(dims)
    if d_max != d_min:
        msg = "索引向量维度不一致: vrag d=%d, symphony d=%d" % (xv.shape[1], xs.shape[1])
        if xf is not None:
            msg += ", full d=%d" % xf.shape[1]
        print(msg, file=sys.stderr)
        sys.exit(1)

    if not args.no_normalize:
        xv = _row_l2_normalize(xv)
        xs = _row_l2_normalize(xs)
        if xf is not None:
            xf = _row_l2_normalize(xf)

    n_v, n_s = xv.shape[0], xs.shape[0]
    blocks = [xv, xs]
    n_f = 0
    if xf is not None:
        n_f = xf.shape[0]
        blocks.append(xf)
    x_all = np.vstack(blocks)
    label_parts = [np.zeros(n_v, dtype=np.int32), np.ones(n_s, dtype=np.int32)]
    if xf is not None:
        label_parts.append(np.full(n_f, 2, dtype=np.int32))
    labels = np.concatenate(label_parts)

    from sklearn.decomposition import PCA
    from sklearn.manifold import TSNE

    if args.method == "pca":
        pca2 = PCA(n_components=2, random_state=args.random_state)
        xy = pca2.fit_transform(x_all)
        evr = 100.0 * float(np.sum(pca2.explained_variance_ratio_))
        title_suffix_zh = "PCA 2D，解释方差比 %.1f%%" % evr
        title_suffix_en = "PCA 2D (explained variance %.1f%%)" % evr
    else:
        pca_dim = min(args.pca_components, x_all.shape[1], x_all.shape[0] - 1)
        if pca_dim < 2:
            print("样本过少，无法进行 PCA 预降维", file=sys.stderr)
            sys.exit(1)
        z = PCA(n_components=pca_dim, random_state=args.random_state).fit_transform(x_all)
        n_samples = z.shape[0]
        if args.perplexity > 0:
            perp = float(args.perplexity)
        else:
            perp = float(min(30.0, max(5.0, (n_samples - 1) / 8.0)))
        max_perp = float(n_samples - 1) / 3.0 - 1e-6
        if perp >= max_perp:
            perp = max(5.0, max_perp * 0.99)
        tsne = TSNE(
            n_components=2,
            perplexity=perp,
            learning_rate=200.0,
            init="pca",
            random_state=args.random_state,
            max_iter=1000,
            verbose=0,
        )
        xy = tsne.fit_transform(z)
        title_suffix_zh = "PCA(%d)+t-SNE，perplexity=%.1f" % (pca_dim, perp)
        title_suffix_en = "PCA(%d)+t-SNE, perplexity=%.1f" % (pca_dim, perp)

    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    cjk = _matplotlib_cjk_font()
    if cjk is not None:
        plt.rcParams["font.sans-serif"] = [cjk]
        plt.rcParams["axes.unicode_minus"] = False
        title_suffix = title_suffix_zh
        lab_v = "Vrag (%d)" % n_v
        lab_s = "Symphony (%d)" % n_s
        lab_f = "Video Space (%d)" % int(n_f*10+2) if n_f > 0 else None
        main_title = "Faiss 向量二维嵌入对比 — " + title_suffix
        xlab, ylab = "第 1 维", "第 2 维"
    else:
        title_suffix = title_suffix_en
        lab_v = "Vrag (n=%d)" % n_v
        lab_s = "Symphony (n=%d)" % n_s
        lab_f = "Video Space (n=%d)" % int(n_f*10+2) if n_f > 0 else None
        main_title = "Faiss 2D embedding — " + title_suffix
        xlab, ylab = "dim-1", "dim-2"

    # vrag 与 full 交换颜色与散点大小 s；alpha 仍按各自语义
    full_s, full_c, full_a = 100, "#4C72B0", 0.35
    vrag_s, vrag_c, vrag_a = 50, "#f1ac19", 0.55
    sym_s, sym_c, sym_a = 5, "#F45454", 1.0

    m0 = labels == 0
    m1 = labels == 1
    m2 = labels == 2
    xlim, ylim = _axis_limits_from_xy(xy)
    fig_w = 5.5 if n_f > 0 else 5.0
    fig_h = 4.3

    def _apply_common_axes(ax_obj, title_text):
        ax_obj.set_xlim(xlim)
        ax_obj.set_ylim(ylim)
        # ax_obj.set_title(title_text)
        ax_obj.legend(loc="upper center", 
            ncol=3,
            markerscale=2,
            columnspacing=0.1, 
            borderpad=0.5,
            handletextpad=0.2,
            bbox_to_anchor=(0.5, 1.12))
        # ax_obj.set_xlabel(xlab)
        # ax_obj.set_ylabel(ylab)
        ax_obj.grid(False)
        ax_obj.set_xticks([])
        ax_obj.set_yticks([])

    out_dir = os.path.dirname(os.path.abspath(args.output))
    if out_dir:
        if not os.path.isdir(out_dir):
            os.makedirs(out_dir)
    out_abs = os.path.abspath(args.output)

    fig, ax = plt.subplots(figsize=(fig_w, fig_h), dpi=args.dpi)
    if n_f > 0:
        ax.scatter(
            xy[m2, 0],
            xy[m2, 1],
            s=full_s,
            c=full_c,
            alpha=full_a,
            linewidths=0,
            label=lab_f,
        )
    ax.scatter(
        xy[m0, 0],
        xy[m0, 1],
        s=vrag_s,
        c=vrag_c,
        alpha=vrag_a,
        linewidths=0,
        label=lab_v,
    )
    ax.scatter(
        xy[m1, 0],
        xy[m1, 1],
        s=sym_s,
        c=sym_c,
        alpha=sym_a,
        linewidths=0,
        label=lab_s,
    )
    _apply_common_axes(ax, main_title)
    # fig.tight_layout(pad=0.0, rect=(0, 0, 1, 0.93))
    fig.tight_layout()
    fig.savefig(args.output)
    plt.close(fig)
    print("已保存: %s" % out_abs)

    if cjk is not None:
        title_v = "仅 vrag — " + title_suffix
        title_s = "仅 symphony_v5 — " + title_suffix
        title_f = "仅视频全量编码 — " + title_suffix
    else:
        title_v = "vrag only — " + title_suffix
        title_s = "symphony_v5 only — " + title_suffix
        title_f = "video full encoding only — " + title_suffix

    side_paths = {
        "vrag": os.path.join(out_dir or ".", "moti_faiss_tsne_vrag.png"),
        "symphony": os.path.join(out_dir or ".", "moti_faiss_tsne_symphony.png"),
        "full": os.path.join(out_dir or ".", "moti_faiss_tsne_full.png"),
    }

    if n_f > 0:
        fig_f, ax_f = plt.subplots(figsize=(fig_w, fig_h), dpi=args.dpi)
        ax_f.scatter(
            xy[m2, 0],
            xy[m2, 1],
            s=full_s,
            c=full_c,
            alpha=full_a,
            linewidths=0,
            label=lab_f,
        )
        _apply_common_axes(ax_f, title_f)
        fig_f.tight_layout()
        fig_f.savefig(side_paths["full"])
        plt.close(fig_f)
        print("已保存: %s" % os.path.abspath(side_paths["full"]))

    fig_v, ax_v = plt.subplots(figsize=(fig_w, fig_h), dpi=args.dpi)
    ax_v.scatter(
        xy[m0, 0],
        xy[m0, 1],
        s=vrag_s,
        c=vrag_c,
        alpha=vrag_a,
        linewidths=0,
        label=lab_v,
    )
    _apply_common_axes(ax_v, title_v)
    fig_v.tight_layout()
    fig_v.savefig(side_paths["vrag"])
    plt.close(fig_v)
    print("已保存: %s" % os.path.abspath(side_paths["vrag"]))

    fig_s, ax_s = plt.subplots(figsize=(fig_w, fig_h), dpi=args.dpi)
    ax_s.scatter(
        xy[m1, 0],
        xy[m1, 1],
        s=sym_s,
        c=sym_c,
        alpha=sym_a,
        linewidths=0,
        label=lab_s,
    )
    _apply_common_axes(ax_s, title_s)
    fig_s.tight_layout()
    fig_s.savefig(side_paths["symphony"])
    plt.close(fig_s)
    print("已保存: %s" % os.path.abspath(side_paths["symphony"]))
