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

os.environ["MIRAGE_DATA"] = "/ifs/jwst/wit/mirage_data/"
os.environ["CRDS_PATH"] = os.path.join(os.path.expandvars('$HOME'), "crds_cache")
os.environ["CDRS_SERVER_URL"]="https://jwst-cdrs.stsci.edu"



def test_niriss_wfss():

    yaml_output_dir = '.'
    simulations_output_dir = '.'

    reffile_defaults = 'crds'
    cosmic_rays = {'library': 'SUNMAX', 'scale': 1.0}
    background = 'medium'
    pav3 = 12.5
    dates = '2022-10-31'
    datatype = 'linear, raw'

    #pointing_file= get_bigdata("jwst/niriss/wfss/niriss_wfss_example.pointing")
    #xml_file= get_bigdata("jwst/niriss/wfss/niriss_wfss_example.xml")
    #catalog_file= get_bigdata("jwst/niriss/wfss/point_sources.cat")

    pointing_file= '/Users/snweke/mirage/examples/wfss_example_data/niriss_wfss_example.pointing'
    xml_file= '/Users/snweke/mirage/examples/wfss_example_data/niriss_wfss_example.xml'
    catalog_file= '/Users/snweke/mirage/examples/wfss_example_data/point_sources.cat'
    catalogs= {'GOODS-S-FIELD':
              {'point_source': catalog_file}}

    yam= yaml_generator.SimInput(xml_file, pointing_file=pointing_file,
                          catalogs=catalogs, cosmic_rays=cosmic_rays,
                          background=background, roll_angle=pav3,
                          dates=dates, reffile_defaults=reffile_defaults,
                          verbose=True, output_dir=yaml_output_dir,
                          simdata_output_dir=simulations_output_dir,
                          datatype=datatype)

    yam.use_linearized_darks = True

    yam.create_inputs()

    print("Debug")


    yaml_files = glob(os.path.join(yam.output_dir,"jw*.yaml"))
    yaml_WFSS_files = []
    yaml_imaging_files = []



    print(yaml_files)

    print(yam.output_dir, "THIS IS YAML.OUTPUTDIR")

    for f in yaml_files:
        my_dict = yaml.safe_load(open(f))
        if my_dict["Inst"]["mode"]=="wfss":
            yaml_WFSS_files.append(f)
        if my_dict["Inst"]["mode"]=="imaging":
            yaml_imaging_files.append(f)

    print("WFSS files:", yaml_WFSS_files)
    print("Imaging files:", len(yaml_imaging_files))


    with open(yaml_WFSS_files[0], 'r') as infile:
        parameters = yaml.load(infile)

        #parameters['Reffiles']['astrometric'] = 'None'
        #parameters['psf_wing_threshold_file'] = 'config'
        #modified_file = f.replace('.yaml', '_mod.yaml')
        #with io.open(modified_file, 'w') as outfile:
        #    yaml.dump(parameters, outfile, default_flow_style=False)

        #    m =imaging_simulator.ImgSim()
        #    m.paramfile = str(modified_file)
        #    m.create()

        print(yaml_WFSS_files[0], "Name of WFSS file")

    for key in parameters:
        for level2_key in parameters[key]:
            print('{}: {}: {}:'.format(key, level2_key, parameters[key][level2_key]))

    for f in yaml_files:
        m = wfss_simulator.WFSSSim(yaml_imaging_files[0], override_dark=None, save_dispersed_seed=True,
                              extrapolate_SED=True, disp_seed_filename=None, source_stamps_file=None,
                              SED_file=None,SED_normalizing_catalog_column=None, SED_dict=None,
                              create_continuum_seds=False)
        m.create()
