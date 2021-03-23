import os
from glob import glob

from mirage import imaging_simulator
from mirage.yaml import yaml_generator
from jwst.pipeline import Detector1Pipeline
from jwst.pipeline import Image2Pipeline

from ci_watson.artifactory_helpers  import (check_url, get_bigdata_root,
    get_bigdata,
    BigdataError,
)


data_root = get_bigdata_root()

import shutil
import pytest
from astropy.io.fits.diff import FITSDiff



os.environ["MIRAGE_DATA"] = "/ifs/jwst/wit/mirage_data/"
os.environ["CRDS_DATA"] = "/Users/snweke/mirage/crds_cache"
os.environ["CRDS_SERVER_URL"] = "https: //jwst-crds.stsci.edu"


xml_file= 'imaging_example_data/example_imaging_program.xml'

xml_file_src= get_bigdata("jwst/nircam/image/example_imaging_program.xml")

pointing_file= 'imaging_example_data/example_imaging_program.pointing'

pointing_file_src= get_bigdata("jwst/nircam/image/example_imaging_program.pointing")

shutil.copy(xml_file_src, "imaging_example_data")
shutil.copy(pointing_file_src, "imaging_example_data")




catalogs= {'GOODS-S-FIELD':
           {'point_source':  'imaging_example_data/ptsrc_catalog.cat'}}
cosmic_rays= {'library':  'SUNMAX', 'scale': 1.0}
background= 'medium'
pav3= 12.5
roll_angle= pav3
dates= '2022-10-31'
reffile_defaults= 'crds'
verbose= True
#output_dir= '.'
output_dir= './output_imaging_data/'
simulation_dir= './imaging_example_data/'
datatype= 'raw'


def test_nircam_imaging(_jail):

    yfiles = run_yaml_generator(xml_file= xml_file,
                                pointing_file= pointing_file,
                                catalogs= catalogs,
                                cosmic_rays= cosmic_rays,
                                background= background,
                                roll_angle= pav3,
                                dates= dates,
                                reffile_defaults= reffile_defaults,
                                verbose= verbose,
                                output_dir= output_dir,
                                simdata_output_dir= output_dir,
                                datatype= datatype)


    uncal_files = create_simulations(yfiles, output_dir)
    print('\n\n uncal files', uncal_files, '\n\n')
    rate_files = [ ]

    for fname in uncal_files:
        result = Detector1Pipeline.call(fname)
        rate_files.append(result)
        name = result.meta.filename.split("uncal.fits")[0]+'rate.fits'
        print('\n\nname', os.path.join(output_dir, name), '\n\n')
        result.save(os.path.join(output_dir, name))

    for fname in rate_files:
        stage2_result = Image2Pipeline.call(fname)[0]
        name = stage2_result.meta.filename
        stage2_result.save(os.path.join(output_dir, name))

    truth_files = glob(os.path.join('truth', '*.fits'))




def _assert_is_same(output_file, truth_file, **fitsdiff_default_kwargs):


    diff = FITSDiff(output_file, truth_file, **fitsdiff_default_kwargs)
    assert diff.identical, diff.report()







#run the calwebb_image2 pipeline
#compare the rate files to truth files
#compare the cal files to truth files
def run_yaml_generator(xml_file,
                       pointing_file,
                       catalogs,
                       cosmic_rays= None,
                       background= None,
                       roll_angle= None,
                       dates= None,
                       reffile_defaults= 'crds',
                       verbose= True,
                       simdata_output_dir= None,
                       output_dir= None,
                       datatype= datatype):

    yam = yaml_generator.SimInput(input_xml= xml_file,
                                      pointing_file= pointing_file,
                                      catalogs= catalogs,
                                      cosmic_rays= cosmic_rays,
                                      background= background,
                                      roll_angle= pav3,
                                      dates= dates,
                                      reffile_defaults= 'crds',
                                      verbose= True,
                                      output_dir= output_dir,
                                      simdata_output_dir= simdata_output_dir,
                                      datatype= datatype)
    yam.create_inputs()
    yfiles = glob(os.path.join(output_dir, 'jw*.yaml'))
    return yfiles

def create_simulations(input_yaml_files, output_dir):
    for fname in input_yaml_files:
        img_sim = imaging_simulator.ImgSim()
        # img_sim.paramfile = yamlfile
        img_sim.paramfile = fname
        img_sim.create()
        # runs `ImgSim` on the input YAML files
        # return all `_uncal.fits` file
    uncal_files = glob(os.path.join(output_dir, "*_uncal.fits"))
    return uncal_files
