import os
from glob import glob

from mirage import imaging_simulator
from mirage import wfss_simulator
from mirage.utils.constants import FLAMBDA_CGS_UNITS, FLAMBDA_MKS_UNITS, FNU_CGS_UNITS
from mirage.yaml import yaml_generator

from astropy.io import fits
import pkg_resources
import yaml

from astropy.io import fits
import astropy.units as u
from astropy.visualization import simple_norm, imshow_norm
import h5py
import numpy as np

os.environ["MIRAGE_DATA"] = "/ifs/jwst/wit/mirage_data/"
os.environ["CRDS_PATH"] = os.path.join(os.path.expandvars('$HOME'), "crds_cache")
os.environ["CDRS_SERVER_URL"]="https://jwst-cdrs.stsci.edu"

TEST_DATA_DIRECTORY = os.path.normpath(os.path.join(pkg_resources.resource_filename('mirage', ''),
                                                    '../examples/wfss_example_data'))
xml_file = os.path.join("/Users/snweke/new_cloned_mirage_repo/mirage/examples/wfss_example_data", 'niriss_wfss_example.xml')
pointing_file = os.path.join("/Users/snweke/new_cloned_mirage_repo/mirage/examples/wfss_example_data", 'niriss_wfss_example.pointing')
catalogs = {'point_source': os.path.join("/Users/snweke/new_cloned_mirage_repo/mirage/examples/wfss_example_data/",'point_sources.cat')}
reffile_defaults = 'crds'
cosmic_rays = {'library': 'SUNMAX', 'scale': 1.0}
background = 'medium'
pav3 = 12.5
dates = '2022-10-31'
datatype = 'linear, raw'

yaml_output_dir = '//Users/snweke/new_cloned_mirage_repo/output_data/output_yaml_files'
simulations_output_dir = '/Users/snweke/new_cloned_mirage_repo/output_data/output_simulated_data'


def niriss_wfss():
	yam = yaml_generator.SimInput(input_xml=xml_file, pointing_file=pointing_file,
                              catalogs=catalogs, cosmic_rays=cosmic_rays,
                              background=background, roll_angle=pav3,
                              dates=dates, reffile_defaults=reffile_defaults,
                              verbose=True, output_dir=yaml_output_dir,
                              simdata_output_dir=simulations_output_dir,
                              datatype=datatype)
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
				print("WFSS files:",len(yaml_WFSS_files))
				print("Imaging files:",len(yaml_imaging_files))

	with open(yaml_WFSS_files[0], 'r') as infile:
		parameters = yaml.load(infile)
		for key in parameters:
			for level2_key in parameters[key]:
				print('{}: {}: {}'.format(key, level2_key, parameters[key][level2_key]))


def run_wfss_simulator():
	m = wfss_simulator.WFSSSim(yaml_WFSS_files[0], override_dark=None,
							save_dispersed_seed=True,
							extrapolate_SED=True, disp_seed_filename=None,
							source_stamps_file=None, SED_file=None,
							SED_normalizing_catalog_column=None, SED_dict=None,
                            create_continuum_seds=True)
	m.create()

	test_yaml_files = ['jw00042001001_01101_00003_nis.yaml', 'jw00042001001_01101_00005_nis.yaml',
                   'jw00042001001_01101_00009_nis.yaml']
	test_yaml_files = [os.path.join(yaml_output_dir, yfile) for yfile in test_yaml_files]
