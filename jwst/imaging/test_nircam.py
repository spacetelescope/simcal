
import os
os.environ["MIRAGE_DATA"] = "/ifs/jwst/wit/mirage_data/"
os.environ["CRDS_DATA"] = "/Users/snweke/mirage/crds_cache"
os.environ["CRDS_SERVER_URL"] = "https://jwst-crds.stsci.edu"



from glob import glob
from scipy.stats import sigmaclip
import numpy as np
from astropy.io import fits


from mirage import imaging_simulator
from mirage.seed_image import catalog_seed_image
from mirage.dark import dark_prep
from mirage.ramp_generator import obs_generator
from mirage.yaml import yaml_generator




catalogs = {'GOODS-S-FIELD': {'point_source':'imaging_example_data/ptsrc_catalog.cat'}}
cosmic_rays = {'library': 'SUNMAX', 'scale': 1.0}
background='medium'
roll_angle=pav3
dates=dates
reffile_defaults=reffile_defaults
verbose=True
output_dir=output_dir
simdata_output_dir=simulation_dir
datatype=datatype



def test_nircam_imaging():
      # - define a specifc xml_file, pointing_file
     xml_file = 'imaging_example_data/example_imaging_program.xml'
     pointing_file = 'imaging_example_data/example_imaging_program.pointing'
     #call run_yaml_generator() (see function below)
     yfiles = run_yaml_generator(xml_file, pointing_file, catalogs=None, cosmic_rays=None,
                                         background=None, roll_angle=None,
                                         dates=None, reffile_defaults=None,
                                         verbose=True, output_dir=None,
                                         simdata_output_dir=None,
                                         datatype='raw')

      #1- call create_simulations() (see function below)
      uncal_files = create_simulations(yfiles)
      # run the calwebb_detector1 pipeline
      rate_files = [ ]
      for fname in uncal_files:
          result = Detector1Pipeline.call(fname)
          rate_files.append(result)
          result.save(result.meta.filename +".fits")

      # - run the calwebb_image2 pipeline
      #- compare the rate files to truth files
     #- compare the cal files to truth files







def run_yaml_generator(xml_file, pointing_file, catalogs=None, cosmic_rays=None,
                                         background=None, roll_angle=None,
                                         dates=None, reffile_defaults=None,
                                         verbose=True, output_dir=None,
                                         simdata_output_dir=None,
                                         datatype='raw'):
    yam = yaml_generator.SimInput(input_xml=xml_file, pointing_file=pointing_file,
                              catalogs=cat_dict, cosmic_rays=cosmic_rays,
                              background=background, roll_angle=pav3,
                              dates=dates, reffile_defaults=reffile_defaults,
                              verbose=True, output_dir=output_dir,
                              simdata_output_dir=simulation_dir,
                              datatype=datatype)
    yam.create_inputs()
    yfiles = glob(os.path.join(output_dir,'jw*.yaml'))
    return yfiles






def create_simulations(input_yaml_files):

     for fname in input_yaml_files:
           img_sim = imaging_simulator.ImgSim()
           img_sim.paramfile = yamlfile
           img_sim.create()
    # runs `ImgSim` on the input YAML files
    # return all `_uncal.fits` file
     uncal_files = glob("*_uncal.fits")
    return uncal_files




    # We want try out the test functions

def run_yaml_generator(xml_file):
	"""" This is to Generate Mirage XML files from APT XML files """
	print(f"DEBUG: my file is {xml_file}")



def create_simulations(input_yaml_files):
	""" Create simulations using Mirage """

	pass

def test_nircam_mirage_pipeline():

	file_xml = os.path.join(
		os.path.dirname(__file__), "data/file.xml"
		)

	assert os.path.exists(file_xml)
	create_simulations(["foo", "bar"])
	run_yaml_generator(file_xml)


def test_mirage_pipeline():
	pass
