"""Functions for checking generated paprer figures against published paper figures"""

import pandas as pd
import numpy as np
import xarray as xr
import scipy.io as sio

from monsoonbench.spatial import get_india_outline

import matplotlib.pyplot as plt
import cartopy.crs as ccrs
from matplotlib.gridspec import GridSpec


def package_csv_data_as_grid(csv_data, models_list, lat, lon):
    ret = []

    empty_grid = np.full((8, 9), np.nan)
    empty_da = xr.DataArray(
        data=empty_grid,
        coords={"lat": lat, "lon": lon},
    )

    for val in ["fair_brier_skill", "fair_rps_skill"]:
        for h in csv_data["horizon"].unique():
            loop_list = []
            h_data = csv_data.loc[csv_data["horizon"] == h]

            for model in models_list:
                model_da = empty_da.copy()
                model_data = h_data.loc[h_data["dataset"] == model]

                for ind, row in model_data.iterrows():
                    val_to_assign = row[val]
                    lat_idx = np.where(lat == row["lat"])[0][0]
                    lon_idx = np.where(lon == row["lon"])[0][0]
                    model_da.values[lat_idx, lon_idx] = val_to_assign

                model_da = model_da.assign_attrs(
                    units="skill_score_percent",
                    model=model,
                    horizon=h,
                    metric=val
                )
                loop_list.append(model_da)

            ret.append(loop_list)  # inside h loop now

    return ret


def compare_fig6_grids(paper_grid, recreated_grid):
    out = {
        "fair_brier_skill": {15: [], 30: []},
        "fair_rps_skill":   {15: [], 30: []}
    }
    
    for i in range(len(paper_grid)):
        for j in range(len(paper_grid[i])):
            da = paper_grid[i][j]
            score = da.attrs["metric"]
            horizon = da.attrs["horizon"]
            model = da.attrs["model"]
            
            diff = da - recreated_grid[i][j]
            diff.attrs.update({
                "metric": score,
                "horizon": horizon,
                "model": model,
                "units": "skill_score_percent"
            })
            
            out[score][horizon].append(diff)
                
    return out


def create_fig6_style_diff_figure(
    diff_dict,
    lon,
    lat,
    models,
    shpfile_path,
    cmap="RdBu_r",
):
    row_configs = [
        ("fair_brier_skill", 15, "(a) Brier Skill Score: 15-day forecast"),
        ("fair_brier_skill", 30, "(b) Brier Skill Score: 30-day forecast"),
        ("fair_rps_skill",   15, "(c) Ranked Probability Skill Score: 15-day forecast"),
        ("fair_rps_skill",   30, "(d) Ranked Probability Skill Score: 30-day forecast"),
    ]

    def get_vlims(metric):
        all_das = diff_dict[metric][15] + diff_dict[metric][30]
        vals = np.concatenate([da.values.flatten() for da in all_das])
        vals = vals[~np.isnan(vals)]
        vmax = np.nanpercentile(np.abs(vals), 95)
        return -vmax, vmax

    vlims = {
        "fair_brier_skill": get_vlims("fair_brier_skill"),
        "fair_rps_skill":   get_vlims("fair_rps_skill"),
    }

    fig = plt.figure(figsize=(8, 8), dpi=200)

    gs = GridSpec(
        4, 4, figure=fig,
        hspace=0.2,
        wspace=0.02,
        left=0.08, right=0.88, top=0.95, bottom=0.08,
        width_ratios=[1, 1, 1, 0.06],
        height_ratios=[1, 1, 1, 1],
    )

    axes = []
    for row in range(4):
        row_axes = []
        for col in range(3):
            ax = fig.add_subplot(gs[row, col], projection=ccrs.PlateCarree())
            row_axes.append(ax)
        axes.append(row_axes)

    india_boundaries = get_india_outline(shpfile_path)

    images_bss  = None
    images_rpss = None

    for row_idx, (metric, horizon, row_label) in enumerate(row_configs):
        vmin, vmax = vlims[metric]
        das = diff_dict[metric][horizon]

        for col_idx, da in enumerate(das):
            ax = axes[row_idx][col_idx]

            im = ax.pcolormesh(
                lon, lat, da.values,
                cmap=cmap, vmin=vmin, vmax=vmax,
                transform=ccrs.PlateCarree(),
            )

            if metric == "fair_brier_skill" and images_bss is None:
                images_bss = im
            if metric == "fair_rps_skill" and images_rpss is None:
                images_rpss = im

            for boundary in india_boundaries:
                ilon, ilat = boundary
                ax.plot(ilon, ilat, color="black", linewidth=0.5,
                        transform=ccrs.PlateCarree())

            model_name = da.attrs.get("model", models[col_idx])
            ax.text(0.97, 0.97, model_name,
                    transform=ax.transAxes,
                    ha="right", va="top",
                    fontsize=6.5, color="black")

            ax.set_xlim([lon[0] - 4, lon[-1] + 2])
            ax.set_ylim([lat[0] - 4, lat[-1] + 4])

            yticks = np.arange(np.ceil(lat[0] / 8) * 8, lat[-1] + 1, 8)
            ax.set_yticks(yticks)
            if col_idx == 0:
                ax.set_yticklabels([f"{int(y)}°N" for y in yticks], fontsize=5)
            else:
                ax.set_yticklabels([])

            xticks = np.arange(np.ceil(lon[0] / 8) * 8, lon[-1] + 1, 8)
            ax.set_xticks(xticks)
            if row_idx == 3:
                ax.set_xticklabels([f"{int(x)}°E" for x in xticks], fontsize=5)
            else:
                ax.set_xticklabels([])

            ax.tick_params(length=2, width=0.5)
            ax.grid(False)
            ax.set_aspect("equal", adjustable="box")

        # Row label above leftmost panel
        pos = axes[row_idx][0].get_position()
        fig.text(
            pos.x0,
            pos.y1 + 0.005,
            row_label,
            ha="left", va="bottom",
            fontsize=6.5,
        )

    # BSS colorbar (rows 0-1)
    pos_top    = axes[0][2].get_position()
    pos_bottom = axes[1][2].get_position()
    total_height = pos_top.y1 - pos_bottom.y0
    half_height  = total_height * 0.5
    center_y     = (pos_top.y1 + pos_bottom.y0) / 2
    cax_bss = fig.add_axes([0.85, center_y - half_height / 2, 0.01, half_height])
    cbar_bss = fig.colorbar(images_bss, cax=cax_bss, orientation="vertical", extend="both")
    cbar_bss.set_label("BSS (%)", fontsize=6, rotation=270, labelpad=8)
    cbar_bss.ax.tick_params(labelsize=5, length=2, width=1)
    cbar_bss.ax.minorticks_off()

    # RPSS colorbar (rows 2-3)
    pos_top    = axes[2][2].get_position()
    pos_bottom = axes[3][2].get_position()
    total_height = pos_top.y1 - pos_bottom.y0
    half_height  = total_height * 0.5
    center_y     = (pos_top.y1 + pos_bottom.y0) / 2
    cax_rpss = fig.add_axes([0.85, center_y - half_height / 2, 0.01, half_height])
    cbar_rpss = fig.colorbar(images_rpss, cax=cax_rpss, orientation="vertical", extend="both")
    cbar_rpss.set_label("RPSS (%)", fontsize=6, rotation=270, labelpad=8)
    cbar_rpss.ax.tick_params(labelsize=5, length=2, width=1)
    cbar_rpss.ax.minorticks_off()

    fig.suptitle("Skill Score Differences (paper - recreated)")

    return fig


def full_fig6_diagnostic(csv_path,
                         gridded_data,
                         shp_file_path,
                         model_lists = ["ifss2s", "fuxis2s", "ngcm"],
                         lon = np.arange(68, 101, 4),
                         lat = np.arange(8, 37, 4)):
    csv_data = pd.read_csv(csv_path)
    raja_fig_6_data_grid = package_csv_data_as_grid(
    csv_data,
    models_list = model_lists,
    lat=lat,
    lon=lon
    )
    
    fig6_diffs = compare_fig6_grids(raja_fig_6_data_grid, gridded_data)

    fig = create_fig6_style_diff_figure(
        diff_dict = fig6_diffs,
        lon=lon,
        lat=lat,
        models=model_lists,
        shpfile_path=shp_file_path,
    )

    return fig6_diffs, fig


def _load_mat_data(file_path, label):
    print(f"Loading {label} data...")
    data = sio.loadmat(file_path)

    print(f"Available variables in {label} file:", [key for key in data.keys() if not key.startswith('__')])

    if 'lon' in data:
        lon = data['lon'].flatten()
        lat = data['lat'].flatten()
    else:
        print(f"Warning: Coordinates not found in {label} MAT file, using default range")
        lon = np.arange(70, 101, 4)
        lat = np.arange(8, 39, 4)

    return {
        'lon': lon,
        'lat': lat,
        'mae_avg': data['mae_avg'],
        'mae_cmz_mean': data['mae_cmz_mean'],
        'std_er': data['std_er'],
        'false_alarm': data['false_alarm'],
        'far_cmz_mean': data['far_cmz_mean'],
        'miss_rate': data['miss_rate'],
        'mr_cmz_mean': data['mr_cmz_mean'],
    }

def load_spatial_data_15_day(file_path):
    return _load_mat_data(file_path, '15-day')

def load_spatial_data_30_day(file_path):
    return _load_mat_data(file_path, '30-day')


def package_gridded_data(mae, far, mr, lat, lon, models):
    """
    Convert (lat, lon, model) arrays into compare_grids format.
    """
    n_models = len(models) # <-- FIX

    return {
        "mae": [mae[:, :, i].T for i in range(n_models)],
        "far": [far[:, :, i].T * 100 for i in range(n_models)],
        "mr":  [mr[:, :, i].T * 100 for i in range(n_models)],
        "lat": lat,
        "lon": lon
    }

def compare_grids(paper_grid, recreated_grid, lat, lon):
    out = { "mae": [], "far": [], "mr": [], "lat": lat, "lon": lon }
    for metric in paper_grid.keys():
        if metric != "lat" and metric != "lon":
            for ind, d_array in enumerate(paper_grid[metric]):
                diff = d_array - recreated_grid[metric][ind]
                out[metric].append(diff)
                
    return out


def create_8_panel_diff_figure(
    diff_dict,
    lon,
    lat,
    models,
    shp_file_path=None,
    metric="mae",
    cmap="RdBu_r",
    symmetric=True,
):
    """
    8-panel (4x2) difference map for a SINGLE metric.
    """

    grids = diff_dict[metric]
    n_models = len(grids)

    if n_models != 8:
        print(f"Warning: expected 8 models, got {n_models}")

    # Slightly taller figure = prevents crowding
    fig = plt.figure(figsize=(6, 10), dpi=300)

    gs = GridSpec(
        4, 3, figure=fig,
        hspace=0.28,   # FIXED (prevents title overlap)
        wspace=-0.2,
        left=0.05, right=0.85, top=0.95, bottom=0.05,
        width_ratios=[1, 1, 0.08]
    )

    # --- create 8 axes ---
    axes = []
    for row in range(4):
        for col in range(2):
            ax = fig.add_subplot(gs[row, col], projection=ccrs.PlateCarree())
            axes.append(ax)

    print(f"Creating {metric.upper()} diff figure...")

    # --- shared color scaling ---
    all_vals = np.concatenate([g.values.flatten() for g in grids])
    all_vals = all_vals[~np.isnan(all_vals)]

    if symmetric:
        abs_vals = np.abs(all_vals)
        nonzero_vals = abs_vals[(abs_vals > 1e-6) & (abs_vals <= 100)]  # valid % range
    
    if len(nonzero_vals) > 0:
        vmax = np.nanpercentile(nonzero_vals, 95)
    else:
        vmax = 1.0  # fallback if truly all zero
    
    vmin = -vmax

    images = []

    # --- plot panels ---
    for i in range(8):
        ax = axes[i]
        diff = grids[i]

        print(f"Panel {i+1}/8: {metric} - {models[i]}")

        im = ax.pcolormesh(
            lon,
            lat,
            diff,
            cmap=cmap,
            vmin=vmin,
            vmax=vmax,
            transform=ccrs.PlateCarree()
        )
        for j in range(len(lat)):
            for k in range(len(lon)):
                val = diff.values[j, k]
                if np.isnan(val):
                    continue
                text_color = 'black' if -0.1 <= val <= 0.1 else 'white'
                ax.text(
                    lon[k], lat[j], f'{val:.2f}',
                    color=text_color,
                    fontsize=3,
                    ha='center', va='center',
                    transform=ccrs.PlateCarree()
                )

        if shp_file_path:
            india_boundaries = get_india_outline(shp_file_path)
            for boundary in india_boundaries:
                india_lon, india_lat = boundary
                ax.plot(india_lon, india_lat, color='black', linewidth=1)

        # FIXED: add padding so titles don't collide
        ax.set_title(models[i], fontsize=7, pad=8)

        ax.set_xlim([lon[0]-4, 100])
        ax.set_ylim([lat[0]-4, lat[-1]+4])

        yticks = np.arange(lat[0]-2, lat[-1]+3, 8)
        yticklabels = [f"{int(y)}°N" if i % 1 == 0 else "" for i, y in enumerate(yticks)]
        ax.set_yticks(yticks)
        ax.set_yticklabels(yticklabels)

        ax.set_aspect("equal", adjustable="box")
        # ax.set_extent(
        #     [lon.min(), lon.max(), lat.min(), lat.max()],
        #     crs=ccrs.PlateCarree()
        # )
        xticks = np.arange(lon[0]-2, lon[-1]+3, 8)
        xticklabels = [f"{int(x)}°E" if i % 1 == 0 else "" for i, x in enumerate(xticks)]
        ax.set_xticks(xticks)
        ax.set_xticklabels(xticklabels)

        ax.grid(False)



        images.append(im)

    # --- colorbar spanning middle rows ---
    row2_ax = axes[2]
    row3_ax = axes[5]

    row2_pos = row2_ax.get_position()
    row3_pos = row3_ax.get_position()

    colorbar_height = row2_pos.y1 - row3_pos.y0
    colorbar_y_start = row3_pos.y0

    cax = fig.add_axes([
        0.87,
        colorbar_y_start,
        0.025,
        colorbar_height
    ])

    cbar = fig.colorbar(images[0], cax=cax, orientation='vertical')
    cbar.set_label(f"{metric.upper()} difference")

    cbar.ax.tick_params(length=2, width=1)
    cbar.ax.minorticks_off()

    plt.tight_layout()

    return fig


def full_spatial_fig_diagnostic(file_path,
                                label,
                                gridded_data,
                                shp_file_path,
                                model_str=['Climatology', 'IFS', 'AIFS',
                                           'FuXi', 'Graphcast', 'GenCast', 'FuXi-S2S', 'NGCM'],
                                ):
    if label not in ("15-day", "30-day"):
        raise ValueError("label must be '15-day' or '30-day'")

    if label == "15-day":
        data_dict = load_spatial_data_15_day(file_path)
    else:
        data_dict = load_spatial_data_30_day(file_path)

    fig_grid = package_gridded_data(
        mae=data_dict["mae_avg"],
        far=data_dict["false_alarm"],
        mr=data_dict["miss_rate"],
        lat=data_dict["lat"],
        lon=data_dict["lon"],
        models=model_str
    )

    diff_grid = compare_grids(fig_grid, gridded_data,
                              lat=data_dict["lat"], lon=data_dict["lon"])

    fig_mae = create_8_panel_diff_figure(
        diff_grid,
        lon=data_dict["lon"],
        lat=data_dict["lat"],
        models=model_str,
        shp_file_path=shp_file_path,
        metric="mae"
    )
    fig_far = create_8_panel_diff_figure(
        diff_grid,
        lon=data_dict["lon"],
        lat=data_dict["lat"],
        models=model_str,
        shp_file_path=shp_file_path,
        metric="far"
    )
    fig_mr = create_8_panel_diff_figure(
        diff_grid,
        lon=data_dict["lon"],
        lat=data_dict["lat"],
        models=model_str,
        shp_file_path=shp_file_path,
        metric="mr"
    )

    return fig_mae, fig_far, fig_mr