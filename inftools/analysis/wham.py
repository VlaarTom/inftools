from typing import Annotated as Atd
import typer
from typer import Option as Opt
import importlib
import importlib.util
import sys
import os
import tomli
from inftools.analysis.Wham_Pcross import run_analysis

def wham(
    toml: Atd[str, typer.Option("-toml", help="The infretis .toml file")] = "infretis.toml",
    data: Atd[str, typer.Option("-data", help="The infretis_data.txt file")] = "infretis_data.txt",
    nskip: Atd[int, typer.Option("-nskip", help="Number of lines to skip in infretis_data.txt")] = 100,
    lamres: Atd[float, typer.Option("-lamres", help="Resolution along the orderparameter, (intf1-intf0)/10)")] = ...,
    nblock: Atd[int, typer.Option("-nblock", case_sensitive=False, help="Minimal number of blocks in the block-error analysis")] = 5,
    folder: Atd[str, typer.Option("-folder", help="Output folder")] = "wham",
    load: Atd[str, typer.Option("-load", help="Input folder")] = "load",
    # for (conditional) free energy (FE) calculation
    fener: Atd[bool, typer.Option("-fener", help="If set, calculate the conditional free energy. See Wham_")] = False,
    sym: Atd[bool, typer.Option("-sym", help="If set, symmetrized free energy will be calculated")] = False,
    xcol: Atd[int, typer.Option("-xcol", help="What column in order.txt to use as x-value when calculating FE")] = 1,
    ycol: Atd[int, typer.Option("-ycol", help="Same as -xcol but for y-value")] = None,    #  in case of 2D TODO ?
    # ranges (for FE)
    minx: Atd[float, typer.Option("-minx", help="Minimum orderparameter value in the x-direction when calculating FE")] = 0.0,
    maxx: Atd[float, typer.Option("-maxx", help="Maximum orderparameter value in the x-direction when calculating FE")] = 100.0,
    miny: Atd[float, typer.Option("-miny", help="Same as -minx but in y-direction")] = None,
    maxy: Atd[float, typer.Option("-maxy", help="Same as -maxx but in y-direction")] = None,
    # about bins (for FE. setnbins will override nbx/nby if True)
    nbx: Atd[int, typer.Option("-nbx", help="Number of bins in x-direction when calculating the free-energy")] = 100,
    nby: Atd[int, typer.Option("-nby", help="Same as -nbx but in y-direction")] = None,
    setnbins: Atd[bool, typer.Option("-setnbins", help="If set to True, nbins is used to compute the bin width")] = True,
    minbx: Atd[float, typer.Option("-minbx", help="Minimum bin edge in the x-direction when calculating FE")] = -40.0,
    maxbx: Atd[float, typer.Option("-maxbx", help="Maximum bin edge in the x-direction when calculating FE")] = 40.0,
    binw: Atd[float, typer.Option("-binw", help="Bin width when calculating FE")] = 0.25, 
    # for permeability calculation
    zmin: Atd[float, typer.Option("-zmin", help="Min range for DeltaZ region in angstrom for permeability calculation.")] = None,
    zmax: Atd[float, typer.Option("-zmax", help="Max range for DeltaZ region in angstrom for permeability calculation.")] = None,
    timestep: Atd[float, typer.Option("-timestep", help="Time step in fs for flux and permeability calculation.")] = ...,
    ):

    """Run Titus0 wham script."""

    inps = {
        "toml": toml,
        "data": data,
        "nskip": nskip,
        "lamres": lamres,
        "nblock": nblock,
        "fener": fener,
        "sym": sym,
        "zmin": zmin,
        "zmax": zmax,
        "timestep": timestep,
        "folder": folder,
        "histo_stuff":{
            "nbx":nbx, "minx":minx, "maxx":maxx, "xcol":xcol,
            "nby":nby, "miny":miny, "maxy":maxy, "ycol":ycol,
            "minbx":minbx, "maxbx":maxbx, "binw":binw, "setnbins":setnbins
            }
    }

    # load input:
    if os.path.isfile(inps["toml"]):
        with open(inps["toml"], mode="rb") as read:
            config = tomli.load(read)
    else:
        print("No toml file, exit.")
        return
    inps["intfs"] = config["simulation"]["interfaces"]
    inps["subcycle"] = config["engine"].get("subcycles", None)

    if "tis_set" in config["simulation"]:
        inps["lm1"] = config["simulation"]["tis_set"].get("lambda_minus_one", None)
    else:
        inps["lm1"] = None

    if inps["lamres"] is None:
        inps["lamres"] = (inps["intfs"][1] - inps["intfs"][0]) / 10

    if inps["fener"]:
        inps["trajdir"] = config["simulation"].get("load_dir", "load")

    run_analysis(inps)
