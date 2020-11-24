
import os


os.environ["MIRAGE_DATA"] = "/ifs/jwst/wit/mirage_data/"
os.environ["CRDS_DATA"] = "/Users/snweke/mirage/crds_cache"
os.environ["CRDS_SERVER_URL"] = "https://jwst-crds.stsci.edu"





# For examining outputs
from glob import glob
from scipy.stats import sigmaclip
import numpy as np
from astropy.io import fits
import matplotlib.pyplot as plt
get_ipython().run_line_magic('matplotlib', 'inline')




# mirage imports
from mirage import imaging_simulator
from mirage.seed_image import catalog_seed_image
from mirage.dark import dark_prep
from mirage.ramp_generator import obs_generator
from mirage.yaml import yaml_generator




xml_file = 'imaging_example_data/example_imaging_program.xml'
pointing_file = 'imaging_example_data/example_imaging_program.pointing'
cat_dict = {'GOODS-S-FIELD': {'point_source':'imaging_example_data/ptsrc_catalog.cat'}}
reffile_defaults = 'crds'
cosmic_rays = {'library': 'SUNMAX', 'scale': 1.0}
background = 'medium'
pav3 = 12.5
dates = '2022-10-31'
output_dir = './imaging_example_data/'
simulation_dir = './imaging_example_data/'
datatype = 'linear, raw'

# Run the yaml generator

def Run_the_yaml_generator( yam = yaml_generator.SimInput(input_xml=xml_file, pointing_file=pointing_file,
                              catalogs=cat_dict, cosmic_rays=cosmic_rays,
                              background=background, roll_angle=pav3,
                              dates=dates, reffile_defaults=reffile_defaults,
                              verbose=True, output_dir=output_dir,
                              simdata_output_dir=simulation_dir,
                              datatype=datatype))
yam.create_inputs()



	def Create_simulated_data(yamlfile):


 	#yfiles = glob(os.path.join(output_dir,'jw*.yaml'))
 		yamlfile = 'imaging_example_data/jw42424001001_01101_00001_nrcb1.yaml'

 		for file in yamlfile:
 			img_sim = imaging_simulator.ImgSim()
 			img_sim.paramfile = yamlfile
 			img_sim.create()

 		return  _uncal.fits


	Create_simulated_data()



Run_the_yaml_generator()



"""


def Examine_the_output():

	def show(array,title,min=0,max=1000):
   		 	plt.figure(figsize=(12,12))
   		 	plt.imshow(array,clim=(min,max))
  		 	plt.title(title)
   			plt.colorbar().set_label('DN$^{-}$/s'):



		show(img_sim.seedimage,'Seed Image',max=20):


Examine_the_output()






 def Examine_linear_output():

    lin_file = 'imaging_example_data/jw42424001001_01101_00001_nrcb1_linear.fits'
	with fits.open(lin_file) as hdulist:
    	linear_data = hdulist['SCI'].data
	print(linear_data.shape)

	show(linear_data[0, 3, :, :], "Final Group", max=250)

Examine_linear_output()






def Examine_the_raw_output(raw_file = 'imaging_example_data/jw42424001001_01101_00001_nrcb1_uncal.fits'):

	with fits.open(raw_file) as hdulist:
		raw_data = hdulist['SCI'].data
	print(raw_data.shape)


	show(raw_data[0, 3, :, :], "Final Group", max=15000)

	show(1. * raw_data[0, 3, :, :] - 1. * raw_data[0, 0, :, :], "Last Minus First Group", max=200)


Examine_the_raw_output()


"""





#Stage 1
def run_pipeline_stage_1("*uncal.fits"):

  def generate_the_seed_image():
  	cat = catalog_seed_image.Catalog_seed()
	cat.paramfile = yamlfile
	cat.make_seed()


	#show(cat.seedimage,'Seed Image',max=20)
	return rate.fits

	generate_the_seed_image()

run_pipeline_stage_1()



#Stage 2
def run_pipeline_stage_2("*uncal.fits"):


	def Prepare_the_dark_current_exposure():

		d = dark_prep.DarkPrep()
		d.paramfile = yamlfile
		d.prepare()

		exptime = d.linDark.header['NGROUPS'] * cat.frametime
		diff = (d.linDark.data[0,-1,:,:] - d.linDark.data[0,0,:,:]) / exptime
		show(diff,'Dark Current Countrate',max=0.1)


	Prepare_the_dark_current_exposure()





	def Create_the_final_exposure():

		obs = obs_generator.Observation()
		obs.linDark = d.prepDark
		obs.seed = cat.seedimage
		obs.segmap = cat.seed_segmap
		obs.seedheader = cat.seedinfo
		obs.paramfile = yamlfile
		obs.create()



		with fits.open(obs.linear_output) as h:
    		lindata = h[1].data
    	header = h[0].header


    	return rate.fits



 	Create_the_final_exposure()


def run_pipeline_stage_2():



"""

def Show_on_a_log_scale():

	exptime = header['EFFINTTM']
	diffdata = (lindata[0,-1,:,:] - lindata[0,0,:,:]) / exptime
	show(diffdata,'Simulated Data',min=0,max=20)



	offset = 2.
	plt.figure(figsize=(12,12))
	plt.imshow(np.log10(diffdata+offset),clim=(0.001,np.log10(80)))
	plt.title('Simulated Data')
	plt.colorbar().set_label('DN$^{-}$/s')




Show_on_a_log_scale()


"""




#Test Function


def test_nircam_imaging(xml_file, pointing_file):

	if xml_file != 0 and pointing_file != 0:

		return Run_the_yaml_generator()
		return run_pipeline_stage_1()
		return run_pipeline_stage_2()



		for every_rate_file in rate.fits:


			for every truth file in truth_files:

				if every rate ==  every_truth_file:

					return true

				else:

					return  -1


	   	for every_cal_file in cal.file:

	   		for every_truth_file in truth_files:

	   			if every_cal_file == truth_file:

	   				return true

	   			else:

	   				return -1


test_nircam_imaging()
