
import os

from glob import glob


from mirage import imaging_simulator
from mirage import wfss_simulator
from mirage.utils.constants import FLAMBDA_CGS_UNITS, FLAMBDA_MKS_UNITS, FNU_CGS_UNITS
from mirage.yaml import yaml_generator


from ci_watson.artifactory_helpers  import (get_bigdata)

import pytest
from astropy.io.fits.diff import FITSDiff

from mirage import imaging_simulator as im

os.environ["MIRAGE_DATA"] = "/ifs/jwst/wit/mirage_data/"
#os.environ["CRDS_PATH"] = os.path.join(os.path.expandvars('$HOME'), "crds_cache")
os.environ["CRDS_DATA"] = "/Users/snweke/mirage/crds_cache"
os.environ["CDRS_SERVER_URL"]="https://jwst-cdrs.stsci.edu"
os.environ['TEST_NIRISS_DATA'] = os.path.join(os.path.dirname(__file__), 'test_data/NIRISS')

TEST_DATA_DIRECTORY = os.path.normpath(os.path.join(pkg_resources.resource_filename('mirage', ''),
                                                    '../examples/wfss_example_data'))



cosmic_rays = {'library': 'SUNMAX', 'scale': 1.0}
background = 'medium'
pav3 = 12.5
roll_angle= pav3
dates = '2022-10-31'
reffile_defaults = 'crds'
verbose=True

#output_dir= '.'
#simulation_dir= '.'
#datatype= 'raw'





def test_niriss_imaging():
    m = im.ImgSim(offline=True)
    m.paramfile = os.path.join(os.path.dirname(__file__), 'test_data/NIRISS/niriss_imaging_example.yaml')
    m.create()





def niriss_wfss(_jail):

	pointing_file = os.path.join(TEST_DATA_DIRECTORY, 'niriss_wfss_example.pointing')

	xml_file = os.path.join(TEST_DATA_DIRECTORY, 'niriss_wfss_example.xml')

	catalogs = {'point_source': os.path.join(TEST_DATA_DIRECTORY,'point_sources.cat')}


	yaml_output_dir = '/where/to/put/yaml/files'

	simulations_output_dir = '/where/to/put/simulated/data'



	yfiles = run_yaml_generator(xml_file,
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


	yaml_WFSS_files = []
	yaml_imaging_files = []


	for f in yaml_files:
		my_dict = yaml.safe_load(open(f))
		if my_dict["Inst"]["mode"]=="wfss":
        yaml_WFSS_files.append(f)
        if my_dict["Inst"]["mode"]=="imaging":
        	yaml_imaging_files.append(f)
        print("WFSS files:",len(yaml_WFSS_files))
        print("Imaging files:",len(yaml_imaging_files))


	for key in parameters:
		parameters = yaml.load(infile)
		for level2_key in parameters[key]:
			print('{}: {}: {}'.format(key, level2_key, parameters[key][level2_key]))



m = wfss_simulator.WFSSSim(yaml_WFSS_files[0], override_dark=None, save_dispersed_seed=True,
                           extrapolate_SED=True, disp_seed_filename=None, source_stamps_file=None,
                           SED_file=None, SED_normalizing_catalog_column=None, SED_dict=None,
                           create_continuum_seds=True)
m.create()





test_yaml_files = ['jw00042001001_01101_00003_nis.yaml', 'jw00042001001_01101_00005_nis.yaml',
                   'jw00042001001_01101_00009_nis.yaml']
test_yaml_files = [os.path.join(yaml_output_dir, yfile) for yfile in test_yaml_files]




disp_seed_image = 'multiple_yaml_input_no_continuua_dispersed_seed_image.fits'
m = wfss_simulator.WFSSSim(test_yaml_files, override_dark=None, save_dispersed_seed=True,
                           extrapolate_SED=True, disp_seed_filename=disp_seed_image, source_stamps_file=None,
                           SED_file=None, SED_normalizing_catalog_column=None, SED_dict=None,
                           create_continuum_seds=False)
m.create()




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


	yam = yaml_generator.SimInput(input_xml=xml_file, pointing_file=pointing_file,
                              catalogs=catalogs, cosmic_rays=cosmic_rays,
                              background=background, roll_angle=pav3,
                              dates=dates, reffile_defaults=reffile_defaults,
                              add_ghosts=ghosts, convolve_ghosts_with_psf=convolve_ghosts,
                              verbose=True, output_dir=yaml_output_dir,
                              simdata_output_dir=simulations_output_dir,
                              datatype=datatype)

	yam.create_inputs()
	yaml_files = glob(os.path.join(yam.output_dir,"jw*.yaml"))
	return yfiles






def create_simulations(input_yaml_files, output_dir):
	for yaml_imaging_file in yaml_imaging_files[0:1]:
		img_sim = imaging_simulator.ImgSim()
		img_sim.paramfile = yaml_imaging_file
		img_sim.create()

	#uncal_files = glob(os.path.join(output_dir, "*_uncal.fits"))
    #return uncal_files

    final_file = glob(os.path.join(yaml_output_dir, 'jw00042001001_01101_00003_nis_uncal.fits'))
    return final_file
