import os
from glob import glob

from mirage import imaging_simulator
from mirage import wfss_simulator
from mirage.utils.constants import FLAMBDA_CGS_UNITS, FLAMBDA_MKS_UNITS, FNU_CGS_UNITS
from mirage.yaml import yaml_generator
from jwst.pipeline import Detector1Pipeline
from ci_watson.artifactory_helpers  import get_bigdata
from astropy.io import fits


import yaml
import io
import astropy.units as u
import pytest


def test_niriss_wfss(_jail):

    yaml_output_dir = '.'
    simulations_output_dir = '.'

    reffile_defaults = 'crds'
    cosmic_rays = {'library': 'SUNMAX', 'scale': 1.0}
    background = 'medium'
    pav3 = 12.5
    dates = '2022-10-31'
    datatype = 'linear, raw'

    pointing_file= get_bigdata("jwst/niriss/wfss/niriss_wfss_example.pointing")
    xml_file= get_bigdata("jwst/niriss/wfss/niriss_wfss_example.xml")
    catalog_file= get_bigdata("jwst/niriss/wfss/point_sources.cat")

    catalogs= {'point_source': catalog_file}

    yam= yaml_generator.SimInput(xml_file, pointing_file=pointing_file,
                          catalogs=catalogs, cosmic_rays=cosmic_rays,
                          background=background, roll_angle=pav3,
                          dates=dates, reffile_defaults=reffile_defaults,
                          verbose=True, output_dir=yaml_output_dir,
                          simdata_output_dir=simulations_output_dir,
                          datatype=datatype)

    yam.use_linearized_darks = True

    yam.create_inputs()

    yaml_files = glob(os.path.join(yam.output_dir,"jw*.yaml"))
    yaml_WFSS_files = []
    yaml_imaging_files = []

    for f in yaml_files:
        my_dict = yaml.safe_load(open(f))
        if my_dict["Inst"]["mode"]=="wfss":
            yaml_WFSS_files.append(f)
        if my_dict["Inst"]["mode"]=="imaging":
            yaml_imaging_files.append(f)

    print("WFSS files:", yaml_WFSS_files)
    print("Imaging files:", len(yaml_imaging_files))

    for f in yaml_WFSS_files:
        wfss_img_sim = wfss_simulator.WFSSSim(f, override_dark=None, save_dispersed_seed=True,
                       extrapolate_SED=True, disp_seed_filename=None, source_stamps_file=None,
                       SED_file=None, SED_normalizing_catalog_column=None, SED_dict=None,
                       create_continuum_seds=True)
        wfss_img_sim.create()

    for yaml_imaging_file in yaml_imaging_files:
        print("Imaging simulation for {}".format(yaml_imaging_file))
        img_sim = imaging_simulator.ImgSim()
        img_sim.paramfile = yaml_imaging_file
        img_sim.create()
