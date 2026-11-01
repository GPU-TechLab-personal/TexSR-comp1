"""Comparison reports for any experiment: columns of images per map, checked
against GT, drawn as sheets.

A *comparison set* is one material, one run, laid out as (layout.py)

    <set dir>/<map>/<column>.png      one image per map and column, all cropped to
                                      the same valid region and the same size
    <set dir>/sheet_<map>.png         the columns side by side (sheets.strip)
    <set dir>/metrics.json            whatever the producer measured (metrics.py)
    <set dir>/montage*.png            maps x columns grids (montage.py)

Producers decide what the columns are and how each is made comparable (the
MatUp baseline: variants/matup/baseline/postprocess.py; a DeepTextures arm adds
its column through a hook). Everything here only needs the images:

  images.py    read / write, 8/16-bit promotion, display conversion, fit_to (crop a
               border, resize to the common side)
  metrics.py   sharpness, corr, PSNR/SSIM of every column against GT
  sheets.py    drawing: a labelled strip, a rows x columns grid, a materials x
               columns table; shared fonts and paddings
  layout.py    the on-disk convention above: newest run, set folders, cell paths
  montage.py   grids and tables read from comparison sets
  overview.py  CLI: one table over every material for one map
"""
