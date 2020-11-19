
import os


os.environ["MIRAGE_DATA"] = "/ifs/jwst/wit/mirage_data/"
os.environ["CRDS_DATA"] = "/Users/snweke/mirage/crds_cache"
os.environ["CRDS_SERVER_URL"] = "https://jwst-crds.stsci.edu"


from glob import glob
from scipy.stats import sigmaclip
import numpy as np
from astropy.io import fits
import matplotlib.pyplot as plt
get_ipython().run_line_magic('matplotlib', 'inline')


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


yam = yaml_generator.SimInput(input_xml=xml_file, pointing_file=pointing_file,
                              catalogs=cat_dict, cosmic_rays=cosmic_rays,
                              background=background, roll_angle=pav3,
                              dates=dates, reffile_defaults=reffile_defaults,
                              verbose=True, output_dir=output_dir,
                              simdata_output_dir=simulation_dir,
                              datatype=datatype)



yam.create_inputs()



yfiles = glob(os.path.join(output_dir,'jw*.yaml'))


yamlfile = 'imaging_example_data/jw42424001001_01101_00001_nrcb1.yaml'


img_sim = imaging_simulator.ImgSim()
img_sim.paramfile = yamlfile
img_sim.create()


def run_yaml_generator(xml_file, pointing_file, catalogs=None, cosmic_rays=None,
                                         background=None, roll_angle=None,
                                         dates=None, reffile_defaults=None,
                                         verbose=True, output_dir=None,
                                         simdata_output_dir=None,
                                         datatype='raw'):





def test_nircam_imaging():
      # define a specifc xml_file, pointing_file
      xml_file = 'imaging_example_data/example_imaging_program.xml'
      pointing_file = 'imaging_example_data/example_imaging_program.pointing'



      #call run_yaml_generator() (see function below)


      def run_yaml_generator(xml_file, pointing_file, datatype='raw'
                                        catalogs=None, cosmic_rays=None,
                                         background=None, roll_angle=None,
                                         dates=None, reffile_defaults=None,
                                         verbose=True, output_dir=None,
                                         simdata_output_dir=None,
                                         ):

                              yam = yaml_generator.SimInput(input_xml=xml_file, pointing_file=pointing_file,
                              catalogs=cat_dict, cosmic_rays=cosmic_rays,
                              background=background, roll_angle=pav3,
                              dates=dates, reffile_defaults=reffile_defaults,
                              verbose=True, output_dir=output_dir,
                              simdata_output_dir=simulation_dir,
                              datatype=datatype)
    yam.create_inputs()

    #yfiles = glob(os.path.join(output_dir,'jw*.yaml'))

    return yfiles

    run_yaml_generator()

















    """""








    parameter_overrides = {'cosmic_rays': cosmic_rays, 'background': background, 'roll_angle': roll_angle,
                               'dates': dates}

        info = {}
        input_xml = input_xml
        pointing_file = pointing_file
        datatype = datatype
        use_JWST_pipeline = use_JWST_pipeline
        observation_list_file = observation_list_file
        verbose = verbose
        output_dir = output_dir
        simdata_output_dir = simdata_output_dir
        if reffile_defaults in ['crds', 'crds_full_name']:
            reffile_defaults = reffile_defaults
        else:
            raise ValueError("reffile_defaults must be 'crds' or 'crds_full_name'")
        reffile_overrides = reffile_overrides

        table_file = None
        use_nonstsci_names = False
        use_linearized_darks = True
        psfwfe = 'predicted'
        psfwfegroup = 0
        resets_bet_ints = 1  # NIRCam should be 1
        psf_paths = None
        expand_catalog_for_segments = False
        dateobs_for_background = dateobs_for_background
        add_psf_wings = True
        offline = offline

        if ((segmap_flux_limit is not None) and (segmap_flux_limit_units is None)):
            raise ValueError("If segmap_flux_limit is provided, segmap_flux_units must also be provided.")

        if segmap_flux_limit is None:
            segmentation_threshold = SEGMENTATION_MIN_SIGNAL_RATE
        if segmap_flux_limit_units is None:
            segmentation_threshold_units = 'ADU/sec'

        # Expand the MIRAGE_DATA environment variable
        datadir = expand_environment_variable(ENV_VAR, offline=offline)

        # Check that CRDS-related environment variables are set correctly
        crds_datadir = crds_tools.env_variables()

        # Get the path to the 'MIRAGE' package
        modpath = pkg_resources.resource_filename('mirage', '')

        set_global_definitions()
        path_defs()

        if (input_xml is not None):
            if observation_list_file is None:
                observation_list_file = os.path.join(output_dir, 'observation_list.yaml')
            apt_xml_dict = get_observation_dict(input_xml, observation_list_file, catalogs,
                                                     verbose=verbose,
                                                     parameter_overrides=parameter_overrides)
        else:
            logger.error('No input xml file provided. Observation dictionary not constructed.')

        reffile_setup()

        def add_catalogs(xml_file = 'imaging_example_data/example_imaging_program.xml',
            pointing_file = 'imaging_example_data/example_imaging_program.pointing',
            cat_dict = {'GOODS-S-FIELD': {'point_source':'imaging_example_data/ptsrc_catalog.cat'}},
            reffile_defaults = 'crds',
            cosmic_rays = {'library': 'SUNMAX', 'scale': 1.0},
            background = 'medium',
            pav3 = 12.5,
            dates = '2022-10-31',
            output_dir = './imaging_example_data/',
            simulation_dir = './imaging_example_data/',
            datatype = 'linear, raw'):

            #Add list(s) of source catalogs to the table containing the
            #observation information

            n_exposures = len(info['Module'])
            info['point_source'] = [None] * n_exposures
            info['galaxyListFile'] = [None] * n_exposures
            info['extended'] = [None] * n_exposures
            info['convolveExtended'] = [False] * n_exposures
            info['movingTarg'] = [None] * n_exposures
            info['movingTargSersic'] = [None] * n_exposures
            info['movingTargExtended'] = [None] * n_exposures
            info['movingTargToTrack'] = [None] * n_exposures

        for i in range(n_exposures):
            if np.int(info['detector'][i][-1]) < 5:
                filtkey = 'ShortFilter'
                pupilkey = 'ShortPupil'
            else:
                filtkey = 'LongFilter'
                pupilkey = 'LongPupil'
            filt = info[filtkey][i]
            pup = info[pupilkey][i]

            if point_source[i] is not None:
                # In here, we assume the user provided a catalog to go with each filter
                # so now we need to find the filter for each entry and generate a list that makes sense
                info['point_source'][i] = os.path.abspath(os.path.expandvars(
                    catalog_match(filt, pup, point_source, 'point source')))
            else:
                info['point_source'][i] = None
            if galaxyListFile[i] is not None:
                info['galaxyListFile'][i] = os.path.abspath(os.path.expandvars(
                    catalog_match(filt, pup, galaxyListFile, 'galaxy')))
            else:
                info['galaxyListFile'][i] = None
            if extended[i] is not None:
                info['extended'][i] = os.path.abspath(os.path.expandvars(
                    catalog_match(filt, pup, extended, 'extended')))
            else:
                info['extended'][i] = None
            if movingTarg[i] is not None:
                info['movingTarg'][i] = os.path.abspath(os.path.expandvars(
                    catalog_match(filt, pup, movingTarg, 'moving point source target')))
            else:
                info['movingTarg'][i] = None
            if movingTargSersic[i] is not None:
                info['movingTargSersic'][i] = os.path.abspath(os.path.expandvars(
                    catalog_match(filt, pup, movingTargSersic, 'moving sersic target')))
            else:
                info['movingTargSersic'][i] = None
            if movingTargExtended[i] is not None:
                info['movingTargExtended'][i] = os.path.abspath(os.path.expandvars(
                    catalog_match(filt, pup, movingTargExtended, 'moving extended target')))
            else:
                info['movingTargExtended'][i] = None
            if movingTargToTrack[i] is not None:
                info['movingTargToTrack'][i] = os.path.abspath(os.path.expandvars(
                    catalog_match(filt, pup, movingTargToTrack, 'non-sidereal moving target')))
            else:
                info['movingTargToTrack'][i] = None
        if convolveExtended is True:
            info['convolveExtended'] = [True] * n_exposures




def add_crds_reffile_names(xml_file = 'imaging_example_data/example_imaging_program.xml',
        pointing_file = 'imaging_example_data/example_imaging_program.pointing',
        cat_dict = {'GOODS-S-FIELD': {'point_source':'imaging_example_data/ptsrc_catalog.cat'}},
        reffile_defaults = 'crds',
        cosmic_rays = {'library': 'SUNMAX', 'scale': 1.0},
        background = 'medium',
        pav3 = 12.5,
        dates = '2022-10-31',
        output_dir = './imaging_example_data/',
        simulation_dir = './imaging_example_data/',
        datatype = 'linear, raw'):
        """Add specific reference file names to info. This should be
        done if reffile_defaults is set to 'crds_full_name'. For each
        instrument configuration and observing mode, query CRDS for the
        best reference files. Identify which exposures in info
        match the observing configuration, and add the reference file
        names to their records in info.
        """
        all_obs_info, unique_obs_info = info_for_all_observations()

        # Add empty placeholders for reference file entries
        empty_col = np.array([' ' * 500] * len(info['Instrument']))
        superbias_arr = deepcopy(empty_col)
        linearity_arr = deepcopy(empty_col)
        saturation_arr = deepcopy(empty_col)
        gain_arr = deepcopy(empty_col)
        distortion_arr = deepcopy(empty_col)
        photom_arr = deepcopy(empty_col)
        ipc_arr = deepcopy(empty_col)
        ipc_invert = np.array([True] * len(info['Instrument']))
        pixelAreaMap_arr = deepcopy(empty_col)
        transmission_arr = deepcopy(empty_col)
        badpixmask_arr = deepcopy(empty_col)
        pixelflat_arr = deepcopy(empty_col)

        # Loop over combinations, create metadata dict, and get reffiles
        for status in unique_obs_info:
            updated_status = deepcopy(status)
            (instrument, detector, filtername, pupilname, readpattern, exptype) = status

            # Make sure NIRISS filter and pupil values are in the correct wheels
            if instrument == 'NIRISS':
                filtername, pupilname = utils.check_niriss_filter(filtername, pupilname)

            # Create metadata dictionary
            date = datetime.date.today().isoformat()
            current_date = datetime.datetime.now()
            time = current_date.time().isoformat()
            status_dict = {'INSTRUME': instrument, 'DETECTOR': detector,
                           'FILTER': filtername, 'PUPIL': pupilname,
                           'READPATT': readpattern, 'EXP_TYPE': exptype,
                           'DATE-OBS': date, 'TIME-OBS': time,
                           'SUBARRAY': 'FULL'}
            if instrument == 'NIRCAM':
                if detector in ['NRCA5', 'NRCB5', 'NRCALONG', 'NRCBLONG', 'A5', 'B5']:
                    status_dict['CHANNEL'] = 'LONG'
                else:
                    status_dict['CHANNEL'] = 'SHORT'
            if instrument == 'FGS':
                if detector in ['G1', 'G2']:
                    detector = detector.replace('G', 'GUIDER')
                    status_dict['DETECTOR'] = detector
                    updated_status = (instrument, detector, filtername, pupilname, readpattern, exptype)

            # Query CRDS
            # Exclude transmission file for now
            files_no_transmission = list(CRDS_FILE_TYPES.values())
            files_no_transmission.remove('transmission')
            reffiles = crds_tools.get_reffiles(status_dict, files_no_transmission,
                                               download=not offline)

            # If the user entered reference files in reffile_defaults
            # use those over what comes from the CRDS query
            if reffile_overrides is not None:
                manual_reffiles = reffiles_from_dict(updated_status)

                for key in manual_reffiles:
                    if manual_reffiles[key] != 'none':
                        if key == 'badpixmask':
                            crds_key = 'mask'
                        elif key == 'pixelflat':
                            crds_key = 'flat'
                        elif key == 'astrometric':
                            crds_key = 'distortion'
                        elif key == 'pixelAreaMap':
                            crds_key = 'area'
                        else:
                            crds_key = key
                        reffiles[crds_key] = manual_reffiles[key]

            # Transmission image file
            # For the moment, this file is retrieved from NIRCAM_GRISM or NIRISS_GRISM
            # Down the road it will become part of CRDS, at which point
            if 'transmission' not in reffiles.keys():
                reffiles['transmission'] = get_transmission_file(status_dict)
                logger.info('Using transmission file: {}'.format(reffiles['transmission']))

            # Check to see if a version of the inverted IPC kernel file
            # exists already in the same directory. If so, use that and
            # avoid having to invert the kernel at run time.
            inverted_file, must_invert = SimInput.inverted_ipc_kernel_check(reffiles['ipc'])
            if not must_invert:
                reffiles['ipc'] = inverted_file
            reffiles['invert_ipc'] = must_invert

            # Identify entries in the original list that use this combination
            match = [i for i, item in enumerate(all_obs_info) if item==status]

            # Populate the reference file names for the matching entries
            superbias_arr[match] = reffiles['superbias']
            linearity_arr[match] = reffiles['linearity']
            saturation_arr[match] = reffiles['saturation']
            gain_arr[match] = reffiles['gain']
            distortion_arr[match] = reffiles['distortion']
            photom_arr[match] = reffiles['photom']
            ipc_arr[match] = reffiles['ipc']
            ipc_invert[match] = reffiles['invert_ipc']
            pixelAreaMap_arr[match] = reffiles['area']
            transmission_arr[match] = reffiles['transmission']
            badpixmask_arr[match] = reffiles['mask']
            pixelflat_arr[match] = reffiles['flat']

        info['superbias'] = list(superbias_arr)
        info['linearity'] = list(linearity_arr)
        info['saturation'] = list(saturation_arr)
        info['gain'] = list(gain_arr)
        info['astrometric'] = list(distortion_arr)
        info['photom'] = list(photom_arr)
        info['ipc'] = list(ipc_arr)
        info['invert_ipc'] = list(ipc_invert)
        info['pixelAreaMap'] = list(pixelAreaMap_arr)
        info['transmission'] = list(transmission_arr)
        info['badpixmask'] = list(badpixmask_arr)
        info['pixelflat'] = list(pixelflat_arr)

def add_reffile_overrides(xml_file = 'imaging_example_data/example_imaging_program.xml',
        pointing_file = 'imaging_example_data/example_imaging_program.pointing',
        cat_dict = {'GOODS-S-FIELD': {'point_source':'imaging_example_data/ptsrc_catalog.cat'}},
        reffile_defaults = 'crds',
        cosmic_rays = {'library': 'SUNMAX', 'scale': 1.0},
        background = 'medium',
        pav3 = 12.5,
        dates = '2022-10-31',
        output_dir = './imaging_example_data/',
        simulation_dir = './imaging_example_data/',
        datatype = 'linear, raw'):
        """If the user provides a nested dictionary in
        reffile_overrides, search through the dictionary, identify
        exposures that match the observing mode and instrument config
        for each dictionary entry, and populate the reference file
        entries in info with the file names from
        reffile_overrides.
        """
        all_obs_info, unique_obs_info = info_for_all_observations()

        # Add empty placeholders for reference file entries
        empty_col = np.array([' ' * 500] * len(info['Instrument']))
        superbias_arr = deepcopy(empty_col)
        linearity_arr = deepcopy(empty_col)
        saturation_arr = deepcopy(empty_col)
        gain_arr = deepcopy(empty_col)
        distortion_arr = deepcopy(empty_col)
        photom_arr = deepcopy(empty_col)
        ipc_arr = deepcopy(empty_col)
        pixelAreaMap_arr = deepcopy(empty_col)
        transmission_arr = deepcopy(empty_col)
        badpixmask_arr = deepcopy(empty_col)
        pixelflat_arr = deepcopy(empty_col)

        # Loop over combinations, create metadata dict, and get reffiles
        for status in unique_obs_info:
            updated_status = deepcopy(status)
            (instrument, detector, filtername, pupilname, readpattern, exptype) = status

            if instrument == 'FGS':
                if detector in ['G1', 'G2']:
                    detector = detector.replace('G', 'GUIDER')
                    updated_status = (instrument, detector, filtername, pupilname, readpattern, exptype)

            # If the user entered reference files in reffile_defaults
            # use those over what comes from the CRDS query
            #sbias, lin, sat, gainfile, dist, ipcfile, pam = reffiles_from_dict(status)
            manual_reffiles = reffiles_from_dict(updated_status)
            for key in manual_reffiles:
                if manual_reffiles[key] == 'none':
                    manual_reffiles[key] = 'crds'

            # Identify entries in the original list that use this combination
            match = [i for i, item in enumerate(all_obs_info) if item==status]

            # Populate the reference file names for the matching entries
            superbias_arr[match] = manual_reffiles['superbias']
            linearity_arr[match] = manual_reffiles['linearity']
            saturation_arr[match] = manual_reffiles['saturation']
            gain_arr[match] = manual_reffiles['gain']
            distortion_arr[match] = manual_reffiles['distortion']
            photom_arr[match] = manual_reffiles['photom']
            ipc_arr[match] = manual_reffiles['ipc']
            pixelAreaMap_arr[match] = manual_reffiles['area']
            transmission_arr[match] = manual_reffiles['transmission']
            badpixmask_arr[match] = manual_reffiles['badpixmask']
            pixelflat_arr[match] = manual_reffiles['pixelflat']

        info['superbias'] = list(superbias_arr)
        info['linearity'] = list(linearity_arr)
        info['saturation'] = list(saturation_arr)
        info['gain'] = list(gain_arr)
        info['astrometric'] = list(distortion_arr)
        info['photom'] = list(photom_arr)
        info['ipc'] = list(ipc_arr)
        info['pixelAreaMap'] = list(pixelAreaMap_arr)
        info['transmission'] = list(transmission_arr)
        info['badpixmask'] = list(badpixmask_arr)
        info['pixelflat'] = list(pixelflat_arr)

"""
def catalog_match(xml_file = 'imaging_example_data/example_imaging_program.xml',
        pointing_file = 'imaging_example_data/example_imaging_program.pointing',
        cat_dict = {'GOODS-S-FIELD': {'point_source':'imaging_example_data/ptsrc_catalog.cat'}},
        reffile_defaults = 'crds',
        cosmic_rays = {'library': 'SUNMAX', 'scale': 1.0},
        background = 'medium',
        pav3 = 12.5,
        dates = '2022-10-31',
        output_dir = './imaging_example_data/',
        simulation_dir = './imaging_example_data/',
        datatype = 'linear, raw', filter, pupil, catalog_list, cattype):

        Given a filter and pupil value, along with a list of input
        catalogs, find the catalog names that contain each filter/
        pupil name.

        Parameters
        ----------
        filter : str
            Name of a filter element
        pupil : str
            Name of a pupil element
        catalog_list : list
            List of catalog filenames
        cattype : str
            Type of catalog in the list.

        Returns
        -------
        match : str
            Name of catalog that contains the name of the
            input filter/pupil element

        if pupil[0].upper() == 'F':
            match = [s for s in catalog_list if pupil.lower() in s.lower()]
            if len(match) == 0:
                no_catalog_match(pupil, cattype)
                return None
            elif len(match) > 1:
                multiple_catalog_match(pupil, cattype, match)
            return match[0]
        else:
            match = [s for s in catalog_list if filter.lower() in s.lower()]
            if len(match) == 0:
                no_catalog_match(filter, cattype)
                return None
            elif len(match) > 1:
                multiple_catalog_match(filter, cattype, match)
            return match[0]

            """

@logging_functions.log_fail
def create_inputs(xml_file = 'imaging_example_data/example_imaging_program.xml',
        pointing_file = 'imaging_example_data/example_imaging_program.pointing',
        cat_dict = {'GOODS-S-FIELD': {'point_source':'imaging_example_data/ptsrc_catalog.cat'}},
        reffile_defaults = 'crds',
        cosmic_rays = {'library': 'SUNMAX', 'scale': 1.0},
        background = 'medium',
        pav3 = 12.5,
        dates = '2022-10-31',
        output_dir = './imaging_example_data/',
        simulation_dir = './imaging_example_data/',
        datatype = 'linear, raw'):
        """Create observation table """
        path_defs()

        if ((input_xml is not None) &
           (pointing_file is not None) &
           (observation_list_file is not None)):

            # Define directories and paths
            indir, infile = os.path.split(input_xml)
            final_file = os.path.join(output_dir,
                                      'Observation_table_for_' + infile +
                                      '_with_yaml_parameters.csv')

            # Read XML file and make observation table
            apt = apt_inputs.AptInput(input_xml=input_xml, pointing_file=pointing_file,
                                      output_dir=output_dir)
            # apt.input_xml = input_xml
            # apt.pointing_file = pointing_file
            apt.observation_list_file = observation_list_file
            apt.apt_xml_dict = apt_xml_dict

            apt.output_dir = output_dir
            apt.create_input_table()
            info = apt.exposure_tab

            # Add start time info to each element.
            # Ignore warnings as astropy.time.Time will give a warning
            # related to unknown leap seconds if the date is too far in
            # the future.
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                make_start_times()

            # Add a list of output yaml names to the dictionary
            make_output_names()

        elif table_file is not None:
            logger.info('Reading table file: {}'.format(table_file))
            info = ascii.read(table_file)
            info = table_to_dict(info)
            final_file = table_file + '_with_yaml_parameters.csv'

        else:
            raise FileNotFoundError(("WARNING. You must include either an ascii table file of observations"
                                     " or xml and pointing files from APT plus the observation list file."
                                     "Aborting."))

        # If reffile_defaults is set to 'crds' then print the
        # string 'crds' in the entry for all reference files. The
        # actual reference file names will then be identified by
        # querying CRDS at run-time. In this way, once a yaml
        # file has been generated, it will effectively always be
        # up to date, even as reference files change.
        if reffile_defaults == 'crds':
            column_data = ['crds'] * len(info['Instrument'])
            info['superbias'] = column_data
            info['linearity'] = column_data
            info['saturation'] = column_data
            info['gain'] = column_data
            info['astrometric'] = column_data
            info['photom'] = column_data
            info['ipc'] = column_data
            info['invert_ipc'] = np.array([True] * len(info['Instrument']))
            info['pixelAreaMap'] = column_data
            info['transmission'] = column_data
            info['badpixmask'] = column_data
            info['pixelflat'] = column_data

            # If the user provided a dictionary of reference files to
            # override some/all of those from CRDS, then enter those
            # into the correct locations in info
            if reffile_overrides is not None:
                add_reffile_overrides()

        # If reffile_defaults is set to 'crds_full_name' then
        # we query CRDS now, and add the reference file names to the
        # yaml files. This allows for easier tracing of what
        # reference files were used to create a particular exposure.
        # (Although that info is always saved in the header of the
        # exposure itxml_file = 'imaging_example_data/example_imaging_program.xml'
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

        elif reffile_defaults == 'crds_full_name':
            add_crds_reffile_names()

        else:
            raise ValueError(("reffile_defaults is not equal to 'crds' "
                              "nor 'crds_full_name'. Unable to proceed."))

        # Get the list of dark current files to use
        darks = []
        lindarks = []
        for instrument, det in zip([s.lower() for s in info['Instrument']], info['detector']):
            instrument = instrument.lower()
            darks.append(get_dark(instrument, det))
            lindarks.append(get_lindark(instrument, det))

        # If linearized darks are to be used, set the darks to None
        if use_linearized_darks:
            info['dark'] = [None] * len(darks)
            info['lindark'] = lindarks
            if set(lindarks) == set([None]):
                raise RuntimeError(("ERROR: Linearized darks requested, but no linearized dark files "
                                    "found. Check: {}").format(os.path.join(os.path.expandvars('$MIRAGE_DATA'), instrument)))
        else:
            info['dark'] = darks
            info['lindark'] = [None] * len(lindarks)
            if set(darks) == set([None]):
                raise RuntimeError(("ERROR: Raw darks requested, but no raw dark files found. "
                                    "Check: {}").format(os.path.join(os.path.expandvars('$MIRAGE_DATA'), instrument)))

        # Add setting describing whether JWST pipeline will be used
        info['use_JWST_pipeline'] = [use_JWST_pipeline] * len(darks)

        # add background rate to the table
        # info['bkgdrate'] = np.array([bkgdrate]*len(info['Mode']))

        # grism entries
        grism_source_image = ['False'] * len(info['Mode'])
        grism_input_only = ['False'] * len(info['Mode'])
        for i in range(len(info['Mode'])):
            if info['Mode'][i].lower() in ['wfss', 'ts_grism']:
                if info['detector'][i] == 'NIS' or info['detector'][i][-1] == '5':
                    grism_source_image[i] = 'True'
                    grism_input_only[i] = 'True'
                # SW detectors shouldn't be wfss
                if info['Instrument'][i] == 'NIRCAM' and info['detector'][i][-1] != '5':
                    if info['Mode'][i] == 'wfss':
                        info['Mode'][i] = 'imaging'
                    elif info['Mode'][i] == 'ts_grism':
                        info['Mode'][i] = 'ts_imaging'
        info['grism_source_image'] = grism_source_image
        info['grism_input_only'] = grism_input_only

        # Grism TSO adjustments. SW detectors should be flagged ts_imaging mode.
        for i in range(len(info['Mode'])):
            if info['Mode'][i].lower() == 'ts_grism':
                if info['detector'][i][-1] != '5':
                    info['Mode'][i] = 'ts_imaging'

        # level-3 associated keywords that are not present in APT file.
        # not quite sure how to populate these
        info['visit_group'] = ['01'] * len(info['Mode'])
        # info['sequence_id'] = ['1'] * len(info['Mode'])
        seq = []
        for par in info['CoordinatedParallel']:
            if par.lower() == 'true':
                seq.append('2')
            if par.lower() == 'false':
                seq.append('1')
        info['sequence_id'] = seq

        # Deal with user-provided PSFs that differ across observations/visits/exposures
        info['psfpath'] = get_psf_path()

        table = Table(info)
        table.write(final_file, format='csv', overwrite=True)
        logger.info('Updated observation table file saved to {}'.format(final_file))

        # Now go through the lists one element at a time
        # and create a yaml file for each.
        yamls = []
        # for i in range(len(detector_labels)):
        for i, instrument in enumerate(info['Instrument']):
            instrument = instrument.lower()
            if instrument not in 'fgs nircam niriss'.split():
                # do not write files for MIRI and NIRSpec
                continue
            file_dict = {}
            for key in info:
                file_dict[key] = info[key][i]

            # break dither number into numbers for primary
            # and subpixel dithers
            tot_dith = np.int(file_dict['dither'])
            # primarytot = np.int(file_dict['PrimaryDithers'])
            primarytot = np.int(file_dict['number_of_dithers'])

            if isinstance(file_dict['SubpixelPositions'], str) and file_dict['SubpixelPositions'].upper() == 'NONE':
                subpixtot = 1
            else:
                try:
                    subpixtot = np.int(file_dict['SubpixelPositions'])
                except:
                    subpixtot = np.int(file_dict['SubpixelPositions'][0])
            primary_dither = np.ceil(1. * tot_dith / subpixtot)
            file_dict['primary_dither_num'] = np.int(primary_dither)
            subpix_dither = (tot_dith-1) % subpixtot
            file_dict['subpix_dither_num'] = subpix_dither + 1
            file_dict['subarray_def_file'] = global_subarray_definition_files[instrument]
            file_dict['readpatt_def_file'] = global_readout_pattern_files[instrument]
            file_dict['crosstalk_file'] = global_crosstalk_files[instrument]
            file_dict['filtpupilcombo_file'] = global_filtpupilcombo_files[instrument]
            file_dict['filter_position_file'] = global_filter_position_files[instrument]
            file_dict['flux_cal_file'] = global_flux_cal_files[instrument]
            file_dict['psf_wing_threshold_file'] = global_psf_wing_threshold_file[instrument]
            fname = write_yaml(file_dict)
            yamls.append(fname)
        yaml_files = yamls

        # Write out summary of all written yaml files
        filenames = [y.split('/')[-1] for y in yamls]
        mosaic_numbers = sorted(list(set([f.split('_')[0] for f in filenames])))
        obs_ids = sorted(list(set([m[7:10] for m in mosaic_numbers])))

        logger.info('\n')

        total_exposures = 0
        for obs in obs_ids:
            visit_list = list(set([m[10:] for m in mosaic_numbers
                                   if m[7:10] == obs]))
            n_visits = len(visit_list)
            activity_list = list(set([vf[17:29] for vf in info['yamlfile']
                                      if vf[7:10] == obs]))
            n_activities = len(activity_list)
            exposure_list = list(set([vf[20:25] for vf in info['yamlfile']
                                      if vf[7:10] == obs]))
            n_exposures = len(exposure_list) * n_activities
            total_exposures += n_exposures
            all_obs_files = [m for m in info['yamlfile'] if m[7:10] == obs]
            total_files = len(all_obs_files)

            obs_id_int = np.array([np.int(ele) for ele in info['ObservationID']])
            obs_indexes = np.where(obs_id_int == np.int(obs))[0]
            obs_entries = np.array(info['ParallelInstrument'])[obs_indexes]
            coord_par = info['CoordinatedParallel'][obs_indexes[0]]
            if coord_par:
                par_indexes = np.where(obs_entries)[0]
                if len(par_indexes) > 0:
                    parallel_instrument = info['Instrument'][obs_indexes[par_indexes[0]]]
                else:
                    parallel_instrument = 'NONE'
            else:
                parallel_instrument = 'NONE'

            # Note that changing the == below to 'is' results in an error
            pri_indexes = np.where(obs_entries == False)[0]
            prime_instrument = info['Instrument'][obs_indexes[pri_indexes[0]]]

            if prime_instrument.upper() == 'NIRCAM':
                module = info['Module'][obs_indexes[pri_indexes[0]]]
                detectors_used = np.array(info['detector'])[obs_indexes[pri_indexes]]
            elif parallel_instrument.upper() == 'NIRCAM':
                module = info['Module'][obs_indexes[par_indexes[0]]]
                detectors_used = np.array(info['detector'])[obs_indexes[par_indexes]]

            if ((prime_instrument.upper() == 'NIRCAM') or (parallel_instrument.upper() == 'NIRCAM')):
                if module == 'ALL':
                    module = 'A and B'
                n_det = len(set(detectors_used))
            else:
                n_det = 1

            if coord_par == 'true':
                instrument_string = '    Prime: {}, Parallel: {}'.format(prime_instrument, parallel_instrument)
            else:
                instrument_string = '    Prime: {}'.format(prime_instrument)

            logger.info('Observation {}:'.format(obs))
            logger.info(instrument_string)
            logger.info('    {} visit(s)'.format(n_visits))
            logger.info('    {} activity(ies)'.format(n_activities))
            #logger.info('    {} exposure(s)'.format(n_exposures))
            if ((prime_instrument.upper() == 'NIRCAM') or (parallel_instrument.upper() == 'NIRCAM')):
                logger.info('    {} NIRCam detector(s) in module {}'.format(n_det, module))
            logger.info('    {} file(s)'.format(total_files))

        # logger.info('\n{} exposures total.'.format(total_exposures))
        logger.info('{} output files written to: {}'.format(len(yamls), output_dir))
        logger.info('Yaml generator complete')
        log_outdir = os.path.dirname(input_xml)
        logging_functions.move_logfile_to_standard_location(input_xml, STANDARD_LOGFILE_NAME,
                                                            yaml_outdir=log_outdir, log_type='yaml_generator')
"""
def create_output_name(xml_file = 'imaging_example_data/example_imaging_program.xml',
        pointing_file = 'imaging_example_data/example_imaging_program.pointing',
        cat_dict = {'GOODS-S-FIELD': {'point_source':'imaging_example_data/ptsrc_catalog.cat'}},
        reffile_defaults = 'crds',
        cosmic_rays = {'library': 'SUNMAX', 'scale': 1.0},
        background = 'medium',
        pav3 = 12.5,
        dates = '2022-10-31',
        output_dir = './imaging_example_data/',
        simulation_dir = './imaging_example_data/',
        datatype = 'linear, raw', input_obj, index=0):
        Put together the JWST formatted fits file name based on observation parameters

        Parameters
        ----------
        input_obj : dict
            Dictionary from apt_inputs giving information for each exposure

        Returns
        -------
        base : str
            JWST formatted filename base (excluding pipeline step suffix and ".fits")

        proposal_id = '{0:05d}'.format(int(input_obj['ProposalID'][index]))
        observation = input_obj['obs_num'][index]
        visit_number = input_obj['visit_num'][index]
        visit_group = input_obj['visit_group'][index]
        parallel_sequence_id = input_obj['sequence_id'][index]
        activity_id = input_obj['act_id'][index]
        exposure = input_obj['exposure'][index]

        base = 'jw{}{}{}_{}{}{}_{}_'.format(proposal_id, observation, visit_number,
                                            visit_group, parallel_sequence_id, activity_id,
                                            exposure)
        return base


        """

def find_ipc_file(xml_file = 'imaging_example_data/example_imaging_program.xml',
        pointing_file = 'imaging_example_data/example_imaging_program.pointing',
        cat_dict = {'GOODS-S-FIELD': {'point_source':'imaging_example_data/ptsrc_catalog.cat'}},
        reffile_defaults = 'crds',
        cosmic_rays = {'library': 'SUNMAX', 'scale': 1.0},
        background = 'medium',
        pav3 = 12.5,
        dates = '2022-10-31',
        output_dir = './imaging_example_data/',
        simulation_dir = './imaging_example_data/',
        datatype = 'linear, raw', inputipc):
        """Given a list of potential IPC kernel files for a given
        detector, select the most appropriate one, and check to see
        whether the kernel needs to be inverted, in order to populate
        the invertIPC field. This is not intended to be terribly smart.
        The first inverted kernel found will be used. If none are found,
        the first kernel will be used and set to be inverted.

        Parameters
        ----------
        inputipc : list
           List of fits files containing IPC kernels for a single detector

        Returns
        -------
        (ipcfile, invstatus) : tup
           ipcfile is the name of the IPC kernel file to use, and invstatus
           lists whether the kernel needs to be inverted or not.
        """
        for ifile in inputipc:
            kernel = fits.getdata(ifile)
            kshape = kernel.shape

            # If kernel is 4 dimensional, extract the 3x3 kernel associated
            # with a single pixel
            if len(kernel.shape) == 4:
                kernel = kernel[:, :, np.int(kshape[2]/2), np.int(kshape[2]/2)]

            if kernel[1, 1] < 1.0:
                return (ifile, False)
        # If no inverted kernel was found, just return the first file
        return (inputipc[0], True)

def get_dark(xml_file = 'imaging_example_data/example_imaging_program.xml',
        pointing_file = 'imaging_example_data/example_imaging_program.pointing',
        cat_dict = {'GOODS-S-FIELD': {'point_source':'imaging_example_data/ptsrc_catalog.cat'}},
        reffile_defaults = 'crds',
        cosmic_rays = {'library': 'SUNMAX', 'scale': 1.0},
        background = 'medium',
        pav3 = 12.5,
        dates = '2022-10-31',
        output_dir = './imaging_example_data/',
        simulation_dir = './imaging_example_data/',
        datatype = 'linear, raw', instrument, detector):
            """Return the name of a dark current file to use as input
        based on the detector being used

        Parameters
        ----------
        detector : str
            Name of detector being used
        Returns
        -------
        files : str
            Name of a dark current file to use for this detector
        """
files = dark_list[instrument][detector]
if len(files) == 1:
    return files[0]
elif len(files) > 1:
    rand_index = np.random.randint(0, len(files) - 1)
    return files[rand_index]
else:
    return None

    def get_lindark(xml_file = 'imaging_example_data/example_imaging_program.xml',
        pointing_file = 'imaging_example_data/example_imaging_program.pointing',
        cat_dict = {'GOODS-S-FIELD': {'point_source':'imaging_example_data/ptsrc_catalog.cat'}},
        reffile_defaults = 'crds',
        cosmic_rays = {'library': 'SUNMAX', 'scale': 1.0},
        background = 'medium',
        pav3 = 12.5,
        dates = '2022-10-31',
        output_dir = './imaging_example_data/',
        simulation_dir = './imaging_example_data/',
        datatype = 'linear, raw', instrument, detector):
        """
        Return the name of a linearized dark current file to
        use as input based on the detector being used

        Parameters
        ----------
        detector : str
            Name of detector being used

        Returns
        -------
        files : str
            Name of a linearized dark current file to use for this detector
        """
        files = lindark_list[instrument][detector]
        if len(files) == 1:
            return files[0]
        elif len(files) > 1:
            rand_index = np.random.randint(0, len(files) - 1)
            return files[rand_index]
        else:
            return None

    def get_readpattern_defs(xml_file = 'imaging_example_data/example_imaging_program.xml',
        pointing_file = 'imaging_example_data/example_imaging_program.pointing',
        cat_dict = {'GOODS-S-FIELD': {'point_source':'imaging_example_data/ptsrc_catalog.cat'}},
        reffile_defaults = 'crds',
        cosmic_rays = {'library': 'SUNMAX', 'scale': 1.0},
        background = 'medium',
        pav3 = 12.5,
        dates = '2022-10-31',
        output_dir = './imaging_example_data/',
        simulation_dir = './imaging_example_data/',
        datatype = 'linear, raw', filename=None):
        """Read in the readpattern definition file and return table.

        Returns
        -------
        tab : obj
            astropy.table.Table containing readpattern definitions
        filename : str
            Path to input file name
        """
        if filename is not None:
            return ascii.read(filename)

        tab = ascii.read(readpatt_def_file)
        return tab

    def get_reffile(xml_file = 'imaging_example_data/example_imaging_program.xml',
        pointing_file = 'imaging_example_data/example_imaging_program.pointing',
        cat_dict = {'GOODS-S-FIELD': {'point_source':'imaging_example_data/ptsrc_catalog.cat'}},
        reffile_defaults = 'crds',
        cosmic_rays = {'library': 'SUNMAX', 'scale': 1.0},
        background = 'medium',
        pav3 = 12.5,
        dates = '2022-10-31',
        output_dir = './imaging_example_data/',
        simulation_dir = './imaging_example_data/',
        datatype = 'linear, raw', refs, detector):
        """
        Return the appropriate reference file for detector
        and given reference file dictionary.

        Parameters
        ----------
        refs : dict
            dictionary in the form of:
             {'A1':'filenamea1.fits', 'A2':'filenamea2.fits'...}
             Containing reference file names
        detector : str
            Name of detector

        Returns
        -------
        refs: str
            Name of reference file appropriate for given detector
        """
        for key in refs:
            if detector in key:
                return refs[key]
        logger.error("WARNING: no file found for detector {} in {}"
              .format(detector, refs))

    def get_subarray_defs(xml_file = 'imaging_example_data/example_imaging_program.xml',
        pointing_file = 'imaging_example_data/example_imaging_program.pointing',
        cat_dict = {'GOODS-S-FIELD': {'point_source':'imaging_example_data/ptsrc_catalog.cat'}},
        reffile_defaults = 'crds',
        cosmic_rays = {'library': 'SUNMAX', 'scale': 1.0},
        background = 'medium',
        pav3 = 12.5,
        dates = '2022-10-31',
        output_dir = './imaging_example_data/',
        simulation_dir = './imaging_example_data/',
        datatype = 'linear, raw'): #filename=None):
        """Read in subarray definition file and return table

        Returns
        -------
        sub : obj
            astropy.table.Table containing subarray definition information
        filename : str
            Path to input file name
        """
        if filename is not None:
            return ascii.read(filename)

        sub = ascii.read(subarray_def_file)
        return sub

    def info_for_all_observations(xml_file = 'imaging_example_data/example_imaging_program.xml',
        pointing_file = 'imaging_example_data/example_imaging_program.pointing',
        cat_dict = {'GOODS-S-FIELD': {'point_source':'imaging_example_data/ptsrc_catalog.cat'}},
        reffile_defaults = 'crds',
        cosmic_rays = {'library': 'SUNMAX', 'scale': 1.0},
        background = 'medium',
        pav3 = 12.5,
        dates = '2022-10-31',
        output_dir = './imaging_example_data/',
        simulation_dir = './imaging_example_data/',
        datatype = 'linear, raw'):
        """For a given dictionary of observation information, pull out the
        information needed by CRDS from each exposure, and save it as a
        tuple. Place the tuples in a list. Also, return a list of only the
        unique combinations of observation information.

        Returns
        -------
        all_combinations : list
            List of tuples. Each tuple contains the instrument, detector,
            filer, pupil, readout pattern, and exposure type for one exposure.

        unique_combinations : list
            List of tuples similar to ``all_combinations``, but containing
            only one copy of each unique tuple.
        """
        # Get all combinations of instrument, detector, filter, exp_type,
        all_combinations = []
        for i in range(len(info['Instrument'])):
            # Get instrument information for the exposure
            instrument = info['Instrument'][i]
            detector = info['detector'][i]
            if instrument == 'NIRCAM':
                detector = 'NRC{}'.format(detector)
                if '5' in detector:
                    filtername = info['LongFilter'][i]
                    pupilname = info['LongPupil'][i]
                    detector = detector.replace('5', 'LONG')
                else:
                    filtername = info['ShortFilter'][i]
                    pupilname = info['ShortPupil'][i]
            elif instrument == 'NIRISS':
                filtername = info['ShortFilter'][i]
                pupilname = info['ShortPupil'][i]
            elif instrument == 'FGS':
                filtername = 'N/A'
                pupilname = 'N/A'
            readpattern = info['ReadoutPattern'][i]

            if instrument == 'NIRCAM':
                exptype = 'NRC_IMAGE'
            elif instrument == 'NIRISS':
                exptype = 'NIS_IMAGE'
            elif instrument == 'FGS':
                exptype = 'FGS_IMAGE'

            entry = (instrument, detector, filtername, pupilname, readpattern, exptype)
            all_combinations.append(entry)
        unique_combinations = list(set(all_combinations))
        return all_combinations, unique_combinations

    @staticmethod
    def inverted_ipc_kernel_check(filename):
        """Given a CRDS-named IPC reference file, check for the
        existence of a file containing the inverse kernel, which
        is what Mirage really needs. This is based solely on filename,
        using the prefix that Mirage prepends to a kernel filename.

        Parameters
        ----------
        filename : str
            Fits file containing IPC correction kernel

        Returns
        -------
        inverted_filename : str
            Name of fits file containing the inverted IPC kernel

        exists : bool
            True if the file already exists, False if not
        """
        dirname, basename = os.path.split(filename)
        inverted_name = os.path.join(dirname, "Kernel_to_add_IPC_effects_from_{}".format(basename))
        return inverted_name, not os.path.isfile(inverted_name)

    def make_output_names(xml_file = 'imaging_example_data/example_imaging_program.xml',
        pointing_file = 'imaging_example_data/example_imaging_program.pointing',
        cat_dict = {'GOODS-S-FIELD': {'point_source':'imaging_example_data/ptsrc_catalog.cat'}},
        reffile_defaults = 'crds',
        cosmic_rays = {'library': 'SUNMAX', 'scale': 1.0},
        background = 'medium',
        pav3 = 12.5,
        dates = '2022-10-31',
        output_dir = './imaging_example_data/',
        simulation_dir = './imaging_example_data/',
        datatype = 'linear, raw'):
        """Create output yaml file names to go with all of the
        entries in the dictionary
        """
        yaml_names = []
        fits_names = []

        if use_nonstsci_names:
            for i in range(len(info['Module'])):
                act = str(info['act_id'][i]).zfill(2)
                if info['Instrument'][i].lower() == 'niriss':
                    det = 'NIS'
                elif info['Instrument'][i].lower() == 'fgs':
                    det = 'FGS'
                else:
                    det = info['detector'][i]
                mode = info['Mode'][i]
                dither = str(info['dither'][i]).zfill(2)

                yaml_names.append(os.path.abspath(os.path.join(output_dir, 'Act{}_{}_{}_Dither{}.yaml'
                                                                            .format(act, det, mode, dither))))
                fits_names.append('Act{}_{}_{}_Dither{}_uncal.fits'.format(act, det, mode, dither))

        else:
            for i in range(len(info['Module'])):
                if info['Instrument'][i].upper() == 'NIRCAM':
                    fulldetector = 'nrc{}'.format(info['detector'][i].lower())
                else:
                    fulldetector = info['detector'][i].lower()
                outfilebase = create_output_name(info, index=i)
                outfile = "{}{}{}".format(outfilebase, fulldetector, '_uncal.fits')
                yamlout = "{}{}{}".format(outfilebase, fulldetector, '.yaml')

                yaml_names.append(yamlout)
                fits_names.append(outfile)

        info['yamlfile'] = yaml_names
        info['outputfits'] = fits_names
        # Table([info['yamlfile']]).pprint()

    def set_global_definitions(xml_file = 'imaging_example_data/example_imaging_program.xml',
        pointing_file = 'imaging_example_data/example_imaging_program.pointing',
        cat_dict = {'GOODS-S-FIELD': {'point_source':'imaging_example_data/ptsrc_catalog.cat'}},
        reffile_defaults = 'crds',
        cosmic_rays = {'library': 'SUNMAX', 'scale': 1.0},
        background = 'medium',
        pav3 = 12.5,
        dates = '2022-10-31',
        output_dir = './imaging_example_data/',
        simulation_dir = './imaging_example_data/',
        datatype = 'linear, raw'):
        """Store the subarray definitions of all supported instruments."""
        # TODO: Investigate how this could be combined with the creation of
        #  configfiles in reffile_setup()

        global_subarray_definitions = {}
        global_readout_patterns = {}
        global_subarray_definition_files = {}
        global_readout_pattern_files = {}

        global_crosstalk_files = {}
        global_filtpupilcombo_files = {}
        global_filter_position_files = {}
        global_flux_cal_files = {}
        global_psf_wing_threshold_file = {}
        global_psfpath = {}
        # global_filter_throughput_files = {} ?

        for instrument in 'niriss fgs nircam miri nirspec'.split():
            if instrument.lower() == 'niriss':
                readout_pattern_file = 'niriss_readout_pattern.txt'
                subarray_def_file = 'niriss_subarrays.list'
                crosstalk_file = 'niriss_xtalk_zeros.txt'
                filtpupilcombo_file = 'niriss_dual_wheel_list.txt'
                filter_position_file = 'niriss_filter_and_pupil_wheel_positions.txt'
                flux_cal_file = 'niriss_zeropoints.list'
                psf_wing_threshold_file = 'niriss_psf_wing_rate_thresholds.txt'
                psfpath = os.path.join(datadir, 'niriss/gridded_psf_library')
            elif instrument.lower() == 'fgs':
                readout_pattern_file = 'guider_readout_pattern.txt'
                subarray_def_file = 'guider_subarrays.list'
                crosstalk_file = 'guider_xtalk_zeros.txt'
                filtpupilcombo_file = 'guider_filter_dummy.list'
                filter_position_file = 'dummy.txt'
                flux_cal_file = 'guider_zeropoints.list'
                psf_wing_threshold_file = 'fgs_psf_wing_rate_thresholds.txt'
                psfpath = os.path.join(datadir, 'fgs/gridded_psf_library')
            elif instrument.lower() == 'nircam':
                readout_pattern_file = 'nircam_read_pattern_definitions.list'
                subarray_def_file = 'NIRCam_subarray_definitions.list'
                crosstalk_file = 'xtalk20150303g0.errorcut.txt'
                filtpupilcombo_file = 'nircam_filter_pupil_pairings.list'
                filter_position_file = 'nircam_filter_and_pupil_wheel_positions.txt'
                flux_cal_file = 'NIRCam_zeropoints.list'
                psf_wing_threshold_file = 'nircam_psf_wing_rate_thresholds.txt'
                psfpath = os.path.join(datadir, 'nircam/gridded_psf_library')
            else:
                readout_pattern_file = 'N/A'
                subarray_def_file = 'N/A'
                crosstalk_file = 'N/A'
                filtpupilcombo_file = 'N/A'
                filter_position_file = 'N/A'
                flux_cal_file = 'N/A'
                psf_wing_threshold_file = 'N/A'
                psfpath = 'N/A'
            if instrument in 'niriss fgs nircam'.split():
                global_subarray_definitions[instrument] = get_subarray_defs(filename=os.path.join(modpath, 'config', subarray_def_file))
                global_readout_patterns[instrument] = get_readpattern_defs(filename=os.path.join(modpath, 'config', readout_pattern_file))
            global_subarray_definition_files[instrument] = os.path.join(modpath, 'config', subarray_def_file)
            global_readout_pattern_files[instrument] = os.path.join(modpath, 'config', readout_pattern_file)
            global_crosstalk_files[instrument] = os.path.join(modpath, 'config', crosstalk_file)
            global_filtpupilcombo_files[instrument] = os.path.join(modpath, 'config', filtpupilcombo_file)
            global_filter_position_files[instrument] = os.path.join(modpath, 'config', filter_position_file)
            global_flux_cal_files[instrument] = os.path.join(modpath, 'config', flux_cal_file)
            global_psf_wing_threshold_file[instrument] = os.path.join(modpath, 'config', psf_wing_threshold_file)
            global_psfpath[instrument] = psfpath

    def lowercase_dict_keys(xml_file = 'imaging_example_data/example_imaging_program.xml',
        pointing_file = 'imaging_example_data/example_imaging_program.pointing',
        cat_dict = {'GOODS-S-FIELD': {'point_source':'imaging_example_data/ptsrc_catalog.cat'}},
        reffile_defaults = 'crds',
        cosmic_rays = {'library': 'SUNMAX', 'scale': 1.0},
        background = 'medium',
        pav3 = 12.5,
        dates = '2022-10-31',
        output_dir = './imaging_example_data/',
        simulation_dir = './imaging_example_data/',
        datatype = 'linear, raw'):
        """To make reference file override dictionary creation easier for
        users, allow the input keys to be case insensitive. Take the user
        input dictionary and translate all the keys to be lower case.
        """
        for key in reffile_overrides:
            if key.lower() != key:
                newkey = key.lower()
                reffile_overrides[newkey] = reffile_overrides.pop(key)
            else:
                newkey = key
            if isinstance(reffile_overrides[newkey], dict):
                for key2 in reffile_overrides[newkey]:
                    if (key2.lower() != key2):
                        newkey2 = key2.lower()
                        reffile_overrides[newkey][newkey2] = reffile_overrides[newkey].pop(key2)
                    else:
                        newkey2 = key2
                    if isinstance(reffile_overrides[newkey][newkey2], dict):
                        for key3 in reffile_overrides[newkey][newkey2]:
                            if (key3.lower() != key3):
                                newkey3 = key3.lower()
                                reffile_overrides[newkey][newkey2][newkey3] = reffile_overrides[newkey][newkey2].pop(key3)
                            else:
                                newkey3 = key3
                            if isinstance(reffile_overrides[newkey][newkey2][newkey3], dict):
                                for key4 in reffile_overrides[newkey][newkey2][newkey3]:
                                    if (key4.lower() != key4):
                                        newkey4 = key4.lower()
                                        reffile_overrides[newkey][newkey2][newkey3][newkey4] = reffile_overrides[newkey][newkey2][newkey3].pop(key4)
                                    else:
                                        newkey4 = key4
                                    if isinstance(reffile_overrides[newkey][newkey2][newkey3][newkey4], dict):
                                        for key5 in reffile_overrides[newkey][newkey2][newkey3][newkey4]:
                                            if (key5.lower() != key5):
                                                newkey5 = key5.lower()
                                                reffile_overrides[newkey][newkey2][newkey3][newkey4][newkey5] = reffile_overrides[newkey][newkey2][newkey3][newkey4].pop(key5)
                                            else:
                                                newkey5 = key5
                                            if isinstance(reffile_overrides[newkey][newkey2][newkey3][newkey4][newkey5], dict):
                                                for key6 in reffile_overrides[newkey][newkey2][newkey3][newkey4][newkey5]:
                                                    if (key6.lower() != key6):
                                                        newkey6 = key6.lower()
                                                        reffile_overrides[newkey][newkey2][newkey3][newkey4][newkey5][newkey6] = reffile_overrides[newkey][newkey2][newkey3][newkey4][newkey5].pop(key6)
                                                    else:
                                                        newkey6 = key6

    def make_start_times(xml_file = 'imaging_example_data/example_imaging_program.xml',
        pointing_file = 'imaging_example_data/example_imaging_program.pointing',
        cat_dict = {'GOODS-S-FIELD': {'point_source':'imaging_example_data/ptsrc_catalog.cat'}},
        reffile_defaults = 'crds',
        cosmic_rays = {'library': 'SUNMAX', 'scale': 1.0},
        background = 'medium',
        pav3 = 12.5,
        dates = '2022-10-31',
        output_dir = './imaging_example_data/',
        simulation_dir = './imaging_example_data/',
        datatype = 'linear, raw'):
        """Create exposure start times for each entry in the observation dictionary."""
        date_obs = []
        time_obs = []
        expstart = []
        nframe = []
        nskip = []
        namp = []

        # choose arbitrary start time for each epoch
        epoch_base_time = '16:44:12'
        epoch_base_time0 = deepcopy(epoch_base_time)

        if 'epoch_start_date' in info.keys():
            epoch_base_date = info['epoch_start_date'][0]
        else:
            epoch_base_date = info['Date'][0]
        base = Time(epoch_base_date + 'T' + epoch_base_time)
        base_date, base_time = base.iso.split()

        # Pick some arbirary overhead values
        act_overhead = 90  # seconds. (filter change)
        visit_overhead = 600  # seconds. (slew)

        # Get visit, activity_id, dither_id info for first exposure
        ditherid = info['dither'][0]
        actid = info['act_id'][0]
        visit = info['visit_num'][0]
        # obsname = info['obs_label'][0]

        # for i in range(len(info['Module'])):
        for i, instrument in enumerate(info['Instrument']):
            # Get dither/visit
            # Files with the same activity_id should have the same start time
            # Overhead after a visit break should be large, smaller between
            # exposures within a visit
            next_actid = info['act_id'][i]
            next_visit = info['visit_num'][i]
            next_obsname = info['obs_label'][i]
            next_ditherid = info['dither'][i]

            # Find the readpattern of the file
            readpatt = info['ReadoutPattern'][i]
            groups = np.int(info['Groups'][i])
            integrations = np.int(info['Integrations'][i])

            if instrument.lower() in ['miri', 'nirspec']:
                nframe.append(0)
                nskip.append(0)
                namp.append(0)
                date_obs.append(base_date)
                time_obs.append(base_time)
                expstart.append(base.mjd)

            else:
                # Now read in readpattern definitions
                readpatt_def = global_readout_patterns[instrument.lower()]

                # Read in file containing subarray definitions
                subarray_def = global_subarray_definitions[instrument.lower()]

                match2 = readpatt == readpatt_def['name']
                if np.sum(match2) == 0:
                    raise RuntimeError(("WARNING!! Readout pattern {} not found in definition file."
                                        .format(readpatt)))

                # Now get nframe and nskip so we know how many frames in a group
                fpg = np.int(readpatt_def['nframe'][match2][0])
                spg = np.int(readpatt_def['nskip'][match2][0])
                nframe.append(fpg)
                nskip.append(spg)

                # Get the aperture name. For non-NIRCam instruments,
                # this is simply the info['aperture']. But for NIRCam,
                # we need to be careful of entries like NRCBS_FULL, which is used
                # for observations using all 4 shortwave B detectors. In that case,
                # we need to build the aperture name from the combination of detector
                # and subarray name.
                aperture = info['aperture'][i]

                # Get the number of amps from the subarray definition file
                match = aperture == subarray_def['AperName']

                # needed for NIRCam case
                if np.sum(match) == 0:
                    aperture = [apername for apername, name in
                                np.array(subarray_def['AperName', 'Name']) if
                                (sub in apername) or (sub in name)]

                    match = aperture == subarray_def['AperName']

                    if len(aperture) > 1 or len(aperture) == 0 or np.sum(match) == 0:
                        raise ValueError('Cannot combine detector {} and subarray {}\
                            into valid aperture name.'.format(det, sub))
                    # We don't want aperture as a list
                    aperture = aperture[0]

                # For grism tso observations, get the number of
                # amplifiers to use from the APT file.
                # For other modes, check the subarray def table.
                try:
                    amp = int(info['NumOutputs'][i])
                except ValueError:
                    amp = subarray_def['num_amps'][match][0]

                # Default to amps=4 for subarrays that can have 1 or 4
                # if the number of amps is not defined. Hopefully we
                # should never enter this code block given the lines above.
                if amp == 0:
                    amp = 4
                    logger.info(('Aperture {} can be used with 1 or 4 readout amplifiers. Defaulting to use 4.'
                                      'In the future this information should be made a user input.'.format(aperture)))
                namp.append(amp)

                # same activity ID
                # Remove this for now, since Mirage was not correctly
                # specifying activities. At the moment all exposures have
                # the same activity ID, which means we must allow the
                # the epoch_start_date to change even if the activity ID
                # does not. This will change back in the future when we
                # figure out more realistic activity ID values.
                #if next_actid == actid:
                #    # in this case, the start time should remain the same
                #    date_obs.append(base_date)
                #    time_obs.append(base_time)
                #    expstart.append(base.mjd)
                #    continue

                epoch_date = info['epoch_start_date'][i]
                epoch_time = deepcopy(epoch_base_time0)

                # new epoch - update the base time
                if epoch_date != epoch_base_date:
                    epoch_base_date = deepcopy(epoch_date)
                    base = Time(epoch_base_date + 'T' + epoch_base_time)
                    base_date, base_time = base.iso.split()
                    basereset = True
                    date_obs.append(base_date)
                    time_obs.append(base_time)
                    expstart.append(base.mjd)
                    actid = deepcopy(next_actid)
                    visit = deepcopy(next_visit)
                    obsname = deepcopy(next_obsname)
                    continue

                # new visit
                if next_visit != visit:
                    # visit break. Larger overhead
                    overhead = visit_overhead

                # This block should be updated when we have more realistic
                # activity IDs
                elif ((next_actid > actid) & (next_visit == visit)):
                    # same visit, new activity. Smaller overhead
                    overhead = act_overhead
                elif ((next_ditherid != ditherid) & (next_visit == visit)):
                    # same visit, new dither position. Smaller overhead
                    overhead = act_overhead
                else:
                    # same observation, activity, dither. Filter changes
                    # will still fall in here, which is not accurate
                    overhead = 10.

                # For cases where the base time needs to change
                # continue down here
                siaf_inst = info['Instrument'][i].upper()
                if siaf_inst == 'NIRCAM':
                    siaf_inst = "NIRCam"
                siaf_obj = pysiaf.Siaf(siaf_inst)[aperture]

                # Calculate the readout time for a single frame
                frametime = calc_frame_time(siaf_inst, aperture,
                                            siaf_obj.XSciSize, siaf_obj.YSciSize, amp)

                # Estimate total exposure time
                exptime = ((fpg + spg) * groups + fpg) * integrations * frametime

                # Delta should include the exposure time, plus overhead
                delta = TimeDelta(exptime + overhead, format='sec')
                base += delta
                base_date, base_time = base.iso.split()

                # Add updated dates and times to the list
                date_obs.append(base_date)
                time_obs.append(base_time)
                expstart.append(base.mjd)

                # increment the activity ID and visit
                actid = deepcopy(next_actid)
                visit = deepcopy(next_visit)
                obsname = deepcopy(next_obsname)
                ditherid = deepcopy(next_ditherid)

        info['date_obs'] = date_obs
        info['time_obs'] = time_obs
        # info['expstart'] = expstart
        info['nframe'] = nframe
        info['nskip'] = nskip
        info['namp'] = namp

    def multiple_catalog_match(xml_file = 'imaging_example_data/example_imaging_program.xml',
        pointing_file = 'imaging_example_data/example_imaging_program.pointing',
        cat_dict = {'GOODS-S-FIELD': {'point_source':'imaging_example_data/ptsrc_catalog.cat'}},
        reffile_defaults = 'crds',
        cosmic_rays = {'library': 'SUNMAX', 'scale': 1.0},
        background = 'medium',
        pav3 = 12.5,
        dates = '2022-10-31',
        output_dir = './imaging_example_data/',
        simulation_dir = './imaging_example_data/',
        datatype = 'linear, raw', filter, cattype, matchlist):
        """
        Alert the user if more than one catalog matches the filter/pupil

        Parameters
        ----------
        filter : str
          Name of filter element
        cattype : str
          Type of catalog (e.g. pointsource)
        matchlist : list
          Matching catalog names
        """
        logger.warning("WARNING: multiple {} catalogs matched! Using the first.".format(cattype))
        logger.warning("Observation filter: {}".format(filter))
        logger.warning("Matched point source catalogs: {}".format(matchlist))

    def no_catalog_match(xml_file = 'imaging_example_data/example_imaging_program.xml',
        pointing_file = 'imaging_example_data/example_imaging_program.pointing',
        cat_dict = {'GOODS-S-FIELD': {'point_source':'imaging_example_data/ptsrc_catalog.cat'}},
        reffile_defaults = 'crds',
        cosmic_rays = {'library': 'SUNMAX', 'scale': 1.0},
        background = 'medium',
        pav3 = 12.5,
        dates = '2022-10-31',
        output_dir = './imaging_example_data/',
        simulation_dir = './imaging_example_data/',
        datatype = 'linear, raw', filter, cattype):
        """
        Alert user if no catalog match was found.

        Parameters
        ----------
        filter : str
          Name of filter element
        cattype : str
          Type of catalog (e.g. pointsource)

        """
        logger.warning("WARNING: unable to find filter ({}) name".format(filter))
        logger.warning("in any of the given {} inputs".format(cattype))
        logger.warning("Using the first input for now. Make sure input catalog names have")
        logger.warning("the appropriate filter name in the filename to get matching to work.")

    def path_defs(xml_file = 'imaging_example_data/example_imaging_program.xml',
        pointing_file = 'imaging_example_data/example_imaging_program.pointing',
        cat_dict = {'GOODS-S-FIELD': {'point_source':'imaging_example_data/ptsrc_catalog.cat'}},
        reffile_defaults = 'crds',
        cosmic_rays = {'library': 'SUNMAX', 'scale': 1.0},
        background = 'medium',
        pav3 = 12.5,
        dates = '2022-10-31',
        output_dir = './imaging_example_data/',
        simulation_dir = './imaging_example_data/',
        datatype = 'linear, raw'):
        """Expand input files to have full paths"""
        if input_xml is not None:
            input_xml = os.path.abspath(os.path.expandvars(input_xml))
        if pointing_file is not None:
            pointing_file = os.path.abspath(os.path.expandvars(pointing_file))
        output_dir = os.path.abspath(os.path.expandvars(output_dir))
        simdata_output_dir = os.path.abspath(os.path.expandvars(simdata_output_dir))
        if table_file is not None:
            table_file = os.path.abspath(os.path.expandvars(table_file))

        ensure_dir_exists(output_dir)
        ensure_dir_exists(simdata_output_dir)

        # subarray_def_file = set_config(subarray_def_file, 'subarray_def_file')
        # readpatt_def_file = set_config(readpatt_def_file, 'readpatt_def_file')
        # filtpupil_pairs = set_config(filtpupil_pairs, 'filtpupil_pairs')
        # fluxcal = set_config(fluxcal, 'fluxcal')
        # filter_throughput = set_config(filter_throughput, 'filter_throughput')
        # dq_init_config = set_config(dq_init_config, 'dq_init_config')
        # refpix_config = set_config(refpix_config, 'refpix_config')
        # saturation_config = set_config(saturation_config, 'saturation_config')
        # superbias_config = set_config(superbias_config, 'superbias_config')
        # linearity_config = set_config(linearity_config, 'linearity_config')

        if observation_list_file is not None:
            observation_list_file = os.path.abspath(os.path.expandvars(observation_list_file))
        # if crosstalk not in [None, 'config']:
        #     crosstalk = os.path.abspath(os.path.expandvars(crosstalk))
        # elif crosstalk == 'config':
        #     crosstalk = os.path.join(modpath, 'config', configfiles['crosstalk'])

    def reffile_setup(xml_file = 'imaging_example_data/example_imaging_program.xml',
        pointing_file = 'imaging_example_data/example_imaging_program.pointing',
        cat_dict = {'GOODS-S-FIELD': {'point_source':'imaging_example_data/ptsrc_catalog.cat'}},
        reffile_defaults = 'crds',
        cosmic_rays = {'library': 'SUNMAX', 'scale': 1.0},
        background = 'medium',
        pav3 = 12.5,
        dates = '2022-10-31',
        output_dir = './imaging_example_data/',
        simulation_dir = './imaging_example_data/',
        datatype = 'linear, raw'):
        """Create lists of reference files associate with each detector.

        Parameters
        ----------
        instrument : str
            Name of instrument
        """
        # Prepare to find files listed as 'config'
        # and set up PSF path

        # set up as dictionary of dictionaries
        configfiles = {}
        psfpath = {}
        psfbasename = {}
        psfpixfrac = {}
        reference_file_dir = {}

        for instrument in 'nircam niriss fgs'.split():
            configfiles[instrument] = {}
            psfpath[instrument] = os.path.join(datadir, instrument, 'gridded_psf_library')
            psfbasename[instrument] = instrument
            reference_file_dir[instrument] = os.path.join(datadir, instrument, 'reference_files')

            # Set instrument-specific file paths
            if instrument == 'nircam':
                psfpixfrac[instrument] = 0.25
            elif instrument == 'niriss':
                psfpixfrac[instrument] = 0.1
            elif instrument == 'fgs':
                psfpixfrac[instrument] = 0.1

            # Set global file paths
            configfiles[instrument]['dq_init_config'] = os.path.join(modpath, 'config', 'dq_init.cfg')
            configfiles[instrument]['saturation_config'] = os.path.join(modpath, 'config', 'saturation.cfg')
            configfiles[instrument]['superbias_config'] = os.path.join(modpath, 'config', 'superbias.cfg')
            configfiles[instrument]['refpix_config'] = os.path.join(modpath, 'config', 'refpix.cfg')
            configfiles[instrument]['linearity_config'] = os.path.join(modpath, 'config', 'linearity.cfg')
            configfiles[instrument]['filter_throughput'] = os.path.join(modpath, 'config', 'placeholder.txt')

        for instrument in 'miri nirspec'.split():
            configfiles[instrument] = {}
            psfpixfrac[instrument] = 0
            psfbasename[instrument] = 'N/A'

        # create empty dictionaries
        list_names = 'superbias linearity gain saturation ipc astrometric photom pam dark lindark'.split()
        for list_name in list_names:
            setattr(xml_file = 'imaging_example_data/example_imaging_program.xml',
            pointing_file = 'imaging_example_data/example_imaging_program.pointing',
            cat_dict = {'GOODS-S-FIELD': {'point_source':'imaging_example_data/ptsrc_catalog.cat'}},
            reffile_defaults = 'crds',
            cosmic_rays = {'library': 'SUNMAX', 'scale': 1.0},
            background = 'medium',
            pav3 = 12.5,
            dates = '2022-10-31',
            output_dir = './imaging_example_data/',
            simulation_dir = './imaging_example_data/',
            datatype = 'linear, raw', '{}_list'.format(list_name), {})

        det_list = {}
        det_list['nircam'] = ['A1', 'A2', 'A3', 'A4', 'A5', 'B1', 'B2', 'B3', 'B4', 'B5']
        det_list['niriss'] = ['NIS']
        det_list['fgs'] = ['G1', 'G2']
        det_list['nirspec'] = ['NRS']
        det_list['miri'] = ['MIR']

        for instrument in 'nircam niriss fgs miri nirspec'.split():
            for list_name in list_names:
                getattr(xml_file = 'imaging_example_data/example_imaging_program.xml',
                pointing_file = 'imaging_example_data/example_imaging_program.pointing',
                cat_dict = {'GOODS-S-FIELD': {'point_source':'imaging_example_data/ptsrc_catalog.cat'}},
                reffile_defaults = 'crds',
                cosmic_rays = {'library': 'SUNMAX', 'scale': 1.0},
                background = 'medium',
                pav3 = 12.5,
                dates = '2022-10-31',
                output_dir = './imaging_example_data/',
                simulation_dir = './imaging_example_data/',
                datatype = 'linear, raw', '{}_list'.format(list_name))[instrument] = {}

            if offline:
                # no access to central store. Set all files to none.
                for list_name in list_names:
                    if list_name in 'dark lindark'.split():
                        default_value = ['None']
                    else:
                        default_value = 'None'
                    for det in det_list[instrument]:
                        getattr(xml_file = 'imaging_example_data/example_imaging_program.xml',
                        pointing_file = 'imaging_example_data/example_imaging_program.pointing',
                        cat_dict = {'GOODS-S-FIELD': {'point_source':'imaging_example_data/ptsrc_catalog.cat'}},
                        reffile_defaults = 'crds',
                        cosmic_rays = {'library': 'SUNMAX', 'scale': 1.0},
                        background = 'medium',
                        pav3 = 12.5,
                        dates = '2022-10-31',
                        output_dir = './imaging_example_data/',
                        simulation_dir = './imaging_example_data/',
                        datatype = 'linear, raw', '{}_list'.format(list_name))[instrument][det] = default_value

            elif instrument == 'nircam':
                rawdark_dir = os.path.join(datadir, 'nircam/darks/raw')
                lindark_dir = os.path.join(datadir, 'nircam/darks/linearized')
                for det in det_list[instrument]:
                    dark_list[instrument][det] = glob(os.path.join(rawdark_dir, det, '*.fits'))
                    lindark_list[instrument][det] = glob(os.path.join(lindark_dir, det, '*.fits'))

            elif instrument in ['nirspec', 'miri']:
                for key in 'subarray_def_file fluxcal filtpupil_pairs readpatt_def_file crosstalk ' \
                           'dq_init_config saturation_config superbias_config refpix_config ' \
                           'linearity_config filter_throughput'.split():
                    configfiles[instrument][key] = 'N/A'
                default_value = 'none'
                for list_name in list_names:
                    for det in det_list[instrument]:
                        getattr(xml_file = 'imaging_example_data/example_imaging_program.xml',
                        pointing_file = 'imaging_example_data/example_imaging_program.pointing',
                        cat_dict = {'GOODS-S-FIELD': {'point_source':'imaging_example_data/ptsrc_catalog.cat'}},
                        reffile_defaults = 'crds',
                        cosmic_rays = {'library': 'SUNMAX', 'scale': 1.0},
                        background = 'medium',
                        pav3 = 12.5,
                        dates = '2022-10-31',
                        output_dir = './imaging_example_data/',
                        simulation_dir = './imaging_example_data/',
                        datatype = 'linear, raw', '{}_list'.format(list_name))[instrument][det] = default_value

            else:  # niriss and fgs
                for det in det_list[instrument]:
                    if det == 'G1':
                        dark_list[instrument][det] = glob(os.path.join(datadir, 'fgs/darks/raw', FGS1_DARK_SEARCH_STRING))
                        lindark_list[instrument][det] = glob(os.path.join(datadir, 'fgs/darks/linearized', FGS1_DARK_SEARCH_STRING))

                    elif det == 'G2':
                        dark_list[instrument][det] = glob(os.path.join(datadir, 'fgs/darks/raw', FGS2_DARK_SEARCH_STRING))
                        lindark_list[instrument][det] = glob(os.path.join(datadir, 'fgs/darks/linearized', FGS2_DARK_SEARCH_STRING))

                    elif det == 'NIS':
                        dark_list[instrument][det] = glob(os.path.join(datadir, 'niriss/darks/raw',
                                                                            '*uncal.fits'))
                        lindark_list[instrument][det] = glob(os.path.join(datadir, 'niriss/darks/linearized',
                                                                               '*linear_dark_prep_object.fits'))

    def set_config(xml_file = 'imaging_example_data/example_imaging_program.xml',
        pointing_file = 'imaging_example_data/example_imaging_program.pointing',
        cat_dict = {'GOODS-S-FIELD': {'point_source':'imaging_example_data/ptsrc_catalog.cat'}},
        reffile_defaults = 'crds',
        cosmic_rays = {'library': 'SUNMAX', 'scale': 1.0},
        background = 'medium',
        pav3 = 12.5,
        dates = '2022-10-31',
        output_dir = './imaging_example_data/',
        simulation_dir = './imaging_example_data/',
        datatype = 'linear, raw', file, prop):
        """
        If a given file is listed as 'config'
        then set it in the yaml output as being in
        the config subdirectory.

        Parameters
        ----------
        file : str
            Name of the input file
        prop : str
            Type of file that file is.

        Returns:
        --------
        file : str
            Full path name to the input file
        """
        if file.lower() not in ['config']:
            file = os.path.abspath(file)
        elif file.lower() == 'config':
            file = os.path.join(modpath, 'config', configfiles[prop])
        return file

    def get_psf_path(xml_file = 'imaging_example_data/example_imaging_program.xml',
        pointing_file = 'imaging_example_data/example_imaging_program.pointing',
        cat_dict = {'GOODS-S-FIELD': {'point_source':'imaging_example_data/ptsrc_catalog.cat'}},
        reffile_defaults = 'crds',
        cosmic_rays = {'library': 'SUNMAX', 'scale': 1.0},
        background = 'medium',
        pav3 = 12.5,
        dates = '2022-10-31',
        output_dir = './imaging_example_data/',
        simulation_dir = './imaging_example_data/',
        datatype = 'linear, raw'):
        """ Create a list of the path to the PSF library directory for
        each observation/visit/exposure in the APT program.

        Parameters:
        -----------
        psf_paths : list, str, or None
            Either a list of the paths to the PSF library(ies), with a
            length equal to the number of activities in the APT program,
            a string containing the path to one PSF library,
            or None. If a list, each path will be written
            chronologically into each yaml file. If a string, that path
            will be written into every yaml file. If None, the
            default PSF library path will be used for all yamls.

        Returns:
        --------
        paths_out : list
            The list of paths to the PSF library(ies), with a length
            equal to the number of activities in the APT program.
        """
        exp_ids = sorted(list(set(info['entry_number'])))
        exp_id_indices = []
        for exp_id in info['entry_number']:
            exp_id_indices.append(exp_ids.index(exp_id))
        n_activities = len(exp_ids)

        # If no path explicitly provided, use the default path.
        if psf_paths is None:
            logger.info('No PSF path provided. Using default path as PSF path for all yamls.')
            paths_out = []
            for instrument in info['Instrument']:
                default_path = global_psfpath[instrument.lower()]
                paths_out.append(default_path)
            return paths_out

        elif isinstance(psf_paths, str):
            logger.info('Using provided PSF path.')
            paths_out = [psf_paths] * len(info['act_id'])
            return paths_out

        elif isinstance(psf_paths, list) and len(psf_paths) != n_activities:
            raise ValueError('Invalid PSF paths parameter provided. Please '
                             'provide the psf_paths in the form of a list of '
                             'strings with a length equal to the number of '
                             'activities in the APT program ({}), not equal to {}.'
                             .format(n_activities, len(psf_paths)))

        elif isinstance(psf_paths, list):
            logger.info('Using provided PSF paths.')
            paths_out = [psf_paths[i] for i in exp_id_indices]
            return paths_out

        elif not isinstance(psf_paths, list) or not isinstance(psf_paths, str):
            raise TypeError('Invalid PSF paths parameter provided. Please '
                            'provide the psf_paths in the form of a list or string, not'
                            '{}'.format(type(psf_paths)))

    def reffiles_from_dict(xml_file = 'imaging_example_data/example_imaging_program.xml',
        pointing_file = 'imaging_example_data/example_imaging_program.pointing',
        cat_dict = {'GOODS-S-FIELD': {'point_source':'imaging_example_data/ptsrc_catalog.cat'}},
        reffile_defaults = 'crds',
        cosmic_rays = {'library': 'SUNMAX', 'scale': 1.0},
        background = 'medium',
        pav3 = 12.5,
        dates = '2022-10-31',
        output_dir = './imaging_example_data/',
        simulation_dir = './imaging_example_data/',
        datatype = 'linear, raw', obs_params):
        """For a given set of observing parameters (instrument, detector,
        filter, etc), check to see if the user has provided any reference
        files to use. These will override the results of the CRDS query.

        Parameters
        ----------
        obs_params : tup
            (instrument, detector, filter, pupil, readpattern, exptype)

        Returns
        -------
        files : dict
            Dictionary containing the reference files that match the
            observing parameters
        """
        files = {}
        (instrument, detector, filtername, pupilname, readpattern, exptype) = obs_params
        instrument = instrument.lower()
        detector = detector.lower()
        filtername = filtername.lower()
        pupilname = pupilname.lower()
        readpattern = readpattern.lower()
        exptype = exptype.lower()

        # Make all of the nested dictionaries such that they have case-
        # insensitive keys. This will allow the user to use upper or lower
        # case for keys in their input dictionaries.
        lowercase_dict_keys()

        # CRDS uses NRCALONG rather than NRCA5.
        if 'long' in detector:
            detector = detector.replace('long', '5')

        # For NIRISS, set filter and pupil names according to where they
        # really are.
        if instrument == 'niriss':
            if filtername.upper() in NIRISS_PUPIL_WHEEL_ELEMENTS:
                temp = deepcopy(filtername)
                filtername = deepcopy(pupilname)
                pupilname = temp
                if pupilname == 'clear':
                    pupilname = 'clearp'

        # superbias
        try:
            if instrument in ['nircam', 'fgs']:
                files['superbias'] = reffile_overrides[instrument]['superbias'][detector][readpattern]
            elif instrument == 'niriss':
                files['superbias'] = reffile_overrides[instrument]['superbias'][readpattern]
        except KeyError:
            files['superbias'] = 'none'

        # linearity
        try:
            if instrument in ['nircam', 'fgs']:
                files['linearity'] = reffile_overrides[instrument]['linearity'][detector]
            elif instrument == 'niriss':
                files['linearity'] = reffile_overrides[instrument]['linearity']
        except KeyError:
            files['linearity'] = 'none'

        # saturation
        try:
            if instrument in ['nircam', 'fgs']:
                files['saturation'] = reffile_overrides[instrument]['saturation'][detector]
            elif instrument == 'niriss':
                files['saturation'] = reffile_overrides[instrument]['saturation']
        except KeyError:
            files['saturation'] = 'none'

        # gain
        try:
            if instrument in ['nircam', 'fgs']:
                files['gain'] = reffile_overrides[instrument]['gain'][detector]
            elif instrument == 'niriss':
                files['gain'] = reffile_overrides[instrument]['gain']
        except KeyError:
            files['gain'] = 'none'

        # distortion
        try:
            if instrument == 'nircam':
                files['distortion'] = reffile_overrides[instrument]['distortion'][detector][filtername][exptype]
            elif instrument == 'niriss':
                files['distortion'] = reffile_overrides[instrument]['distortion'][pupilname][exptype]
            elif instrument == 'fgs':
                files['distortion'] = reffile_overrides[instrument]['distortion'][detector][exptype]
        except KeyError:
            files['distortion'] = 'none'

        # ipc
        try:
            if instrument in ['nircam', 'fgs']:
                files['ipc'] = reffile_overrides[instrument]['ipc'][detector]
            elif instrument == 'niriss':
                files['ipc'] = reffile_overrides[instrument]['ipc']
        except KeyError:
            files['ipc'] = 'none'

        # pixel area map
        try:
            if instrument == 'nircam':
                files['area'] = reffile_overrides[instrument]['area'][detector][filtername][pupilname][exptype]
            elif instrument == 'niriss':
                files['area'] = reffile_overrides[instrument]['area'][filtername][pupilname][exptype]
            elif instrument == 'fgs':
                files['area'] = reffile_overrides[instrument]['area'][detector]
        except KeyError:
            files['area'] = 'none'

        # transmission image
        try:
            if instrument == 'nircam':
                files['transmission'] = reffile_overrides[instrument]['transmission'][detector][filtername][pupilname]
            elif instrument == 'niriss':
                files['transmission'] = reffile_overrides[instrument]['transmission'][filtername][pupilname]
            elif instrument == 'fgs':
                files['transmission'] = reffile_overrides[instrument]['transmission'][detector]
        except KeyError:
            files['transmission'] = 'none'

        # bad pixel map
        try:
            if instrument == 'nircam':
                files['badpixmask'] = reffile_overrides[instrument]['badpixmask'][detector]
            elif instrument == 'niriss':
                files['badpixmask'] = reffile_overrides[instrument]['badpixmask']
            elif instrument == 'fgs':
                files['badpixmask'] = reffile_overrides[instrument]['badpixmask'][detector][exptype]
        except KeyError:
            files['badpixmask'] = 'none'

        # flat field file
        try:
            if instrument == 'nircam':
                files['pixelflat'] = reffile_overrides[instrument]['pixelflat'][detector][filtername][pupilname]
            elif instrument == 'niriss':
                files['pixelflat'] = reffile_overrides[instrument]['pixelflat'][filtername][pupilname]
            elif instrument == 'fgs':
                files['pixelflat'] = reffile_overrides[instrument]['pixelflat'][detector][exptype]
        except KeyError:
            files['pixelflat'] = 'none'

        # photom reference file
        try:
            files['photom'] = reffile_overrides[instrument]['photom'][detector]
        except KeyError:
            files['photom'] = 'none'

        return files

    def table_to_dict(xml_file = 'imaging_example_data/example_imaging_program.xml',
        pointing_file = 'imaging_example_data/example_imaging_program.pointing',
        cat_dict = {'GOODS-S-FIELD': {'point_source':'imaging_example_data/ptsrc_catalog.cat'}},
        reffile_defaults = 'crds',
        cosmic_rays = {'library': 'SUNMAX', 'scale': 1.0},
        background = 'medium',
        pav3 = 12.5,
        dates = '2022-10-31',
        output_dir = './imaging_example_data/',
        simulation_dir = './imaging_example_data/',
        datatype = 'linear, raw', tab):
        """
        Convert the ascii table of observations to a dictionary

        Parameters
        ----------
        tab : obj
            astropy.table.Table containing observation information

        Returns
        -------
        dict : dict
            Dictionary of observation information
        """
        dict = {}
        for colname in tab.colnames:
            dict[colname] = tab[colname].data
        return dict

    def write_yaml(xml_file = 'imaging_example_data/example_imaging_program.xml',
        pointing_file = 'imaging_example_data/example_imaging_program.pointing',
        cat_dict = {'GOODS-S-FIELD': {'point_source':'imaging_example_data/ptsrc_catalog.cat'}},
        reffile_defaults = 'crds',
        cosmic_rays = {'library': 'SUNMAX', 'scale': 1.0},
        background = 'medium',
        pav3 = 12.5,
        dates = '2022-10-31',
        output_dir = './imaging_example_data/',
        simulation_dir = './imaging_example_data/',
        datatype = 'linear, raw', input):
        """
        Create yaml file for a single exposure/detector

        Parameters
        ----------
        input : dict
            dictionary containing all needed exposure
            information for one exposure
        """
        instrument = input['Instrument']
        # select the right filter
        if input['detector'] in ['NIS']:
            # if input['APTTemplate'] == 'NirissExternalCalibration': 'NirissImaging':
            filtkey = 'FilterWheel'
            pupilkey = 'PupilWheel'
            # set the FilterWheel and PupilWheel for NIRISS
            if input['APTTemplate'] in ['NirissAmi']:
                filter_name = input['Filter']
                input[filtkey] = filter_name
                input[pupilkey] = 'NRM'
            elif input['APTTemplate'] not in ['NirissExternalCalibration', 'NirissWfss']:
                filter_name = input['Filter']
                if filter_name in NIRISS_PUPIL_WHEEL_ELEMENTS:
                    input[pupilkey] = filter_name
                    input[filtkey] = 'CLEAR'
                elif filter_name in NIRISS_FILTER_WHEEL_ELEMENTS:
                    input[pupilkey] = 'CLEARP'
                    input[filtkey] = filter_name
                else:
                    raise RuntimeError('Filter {} not valid'.format(filter_name))
            catkey = ''
        elif input['detector'] in ['FGS']:
            filtkey = 'FilterWheel'
            pupilkey = 'PupilWheel'
            catkey = ''
        elif input['detector'] in ['NRS', 'MIR']:
            filtkey = 'FilterWheel'
            pupilkey = 'PupilWheel'
            catkey = ''
        else:
            if np.int(input['detector'][-1]) < 5:
                filtkey = 'ShortFilter'
                pupilkey = 'ShortPupil'
                catkey = 'sw'
            else:
                filtkey = 'LongFilter'
                pupilkey = 'LongPupil'
                catkey = 'lw'

        outfile = input['outputfits']
        yamlout = input['yamlfile']

        yamlout = os.path.join(output_dir, yamlout)
        with open(yamlout, 'w') as f:
            f.write('Inst:\n')
            f.write('  instrument: {}          # Instrument name\n'.format(instrument))
            f.write('  mode: {}                # Observation mode (e.g. imaging, WFSS)\n'.format(input['Mode']))
            f.write('  use_JWST_pipeline: {}   # Use pipeline in data transformations\n'.format(input['use_JWST_pipeline']))
            f.write('\n')
            f.write('Readout:\n')
            f.write('  readpatt: {}        # Readout pattern (RAPID, BRIGHT2, etc) overrides nframe, nskip unless it is not recognized\n'.format(input['ReadoutPattern']))
            f.write('  ngroup: {}              # Number of groups in integration\n'.format(input['Groups']))
            f.write('  nint: {}          # Number of integrations per exposure\n'.format(input['Integrations']))
            f.write('  namp: {}          # Number of amplifiers used to read out detector\n'.format(input['namp']))
            f.write('  resets_bet_ints: {} #Number of detector resets between integrations\n'.format(resets_bet_ints))

            if instrument.lower() == 'nircam':
                # if input['aperture'] in ['NRCA3_DHSPIL', 'NRCB4_DHSPIL']:
                if 'NRCA3_DHSPIL' in input['aperture'] or 'NRCB4_DHSPIL' in input['aperture']: # in ['NRCA3_DHSPIL', 'NRCB4_DHSPIL']:
                    full_ap = input['aperture']
                else:
                    apunder = input['aperture'].find('_')
                    full_ap = 'NRC' + input['detector'] + '_' + input['aperture'][apunder + 1:]
            if instrument.lower() in ['niriss', 'fgs']:
                full_ap = input['aperture']

            subarray_definitions = global_subarray_definitions[instrument.lower()]


            if full_ap not in subarray_definitions['AperName']:
                full_ap_new = [apername for apername, name in
                               np.array(subarray_definitions['AperName', 'Name']) if
                               (full_ap in apername) or (full_ap in name)]
                if len(full_ap_new) > 1 or len(full_ap_new) == 0:
                    raise ValueError('Cannot match {} with valid aperture name for observation {}.'
                                     .format(full_ap, input['obs_num']))
                else:
                    full_ap = full_ap_new[0]

            f.write('  array_name: {}    # Name of array (FULL, SUB160, SUB64P, etc) overrides subarray_bounds below\n'.format(full_ap))
            f.write('  filter: {}       # Filter of simulated data (F090W, F322W2, etc)\n'.format(input[filtkey]))
            f.write('  pupil: {}        # Pupil element for simulated data (CLEAR, GRISMC, etc)\n'.format(input[pupilkey]))
            f.write('\n')
            f.write('Reffiles:                                 # Set to None or leave blank if you wish to skip that step\n')
            f.write('  dark: {}   # Dark current integration used as the base\n'.format(input['dark']))
            f.write('  linearized_darkfile: {}   # Linearized dark ramp to use as input. Supercedes dark above\n'.format(input['lindark']))
            f.write('  badpixmask: {}   # If linearized dark is used, populate output DQ extensions using this file\n'.format(input['badpixmask']))
            f.write('  superbias: {}     # Superbias file. Set to None or leave blank if not using\n'.format(input['superbias']))
            f.write('  linearity: {}    # linearity correction coefficients\n'.format(input['linearity']))
            f.write('  saturation: {}    # well depth reference files\n'.format(input['saturation']))
            f.write('  gain: {} # Gain map\n'.format(input['gain']))
            f.write('  pixelflat: {}    # Flat field file to use for un-flattening output\n'.format(input['pixelflat']))
            f.write('  illumflat: None                               # Illumination flat field file\n')
            f.write('  astrometric: {}  # Astrometric distortion file (asdf)\n'.format(input['astrometric']))
            f.write('  photom: {}   # cal pipeline photom reference file\n'.format(input['photom']))
            f.write('  ipc: {} # File containing IPC kernel to apply\n'.format(input['ipc']))
            f.write(('  invertIPC: {}      # Invert the IPC kernel before the convolution. True or False. Use True if the kernel is '
                     'designed for the removal of IPC effects, like the JWST reference files are.\n'.format(input['invert_ipc'])))
            f.write('  occult: None                                    # Occulting spots correction image\n')
            f.write(('  pixelAreaMap: {}      # Pixel area map for the detector. Used to introduce distortion into the output ramp.\n'
                     .format(input['pixelAreaMap'])))
            f.write(('  transmission: {}      # Transmission image containing fractional throughput map. (e.g. to imprint occulters into fov\n'
                     .format(input['transmission'])))
            f.write(('  subarray_defs: {} # File that contains a list of all possible subarray names and coordinates\n'
                     .format(input['subarray_def_file'])))
            f.write(('  readpattdefs: {}  # File that contains a list of all possible readout pattern names and associated '
                     'NFRAME/NSKIP values\n'.format(input['readpatt_def_file'])))
            f.write('  crosstalk: {}   # File containing crosstalk coefficients\n'.format(input['crosstalk_file']))
            f.write(('  filtpupilcombo: {}   # File that lists the filter wheel element / pupil wheel element combinations. '
                     'Used only in writing output file\n'.format(input['filtpupilcombo_file'])))
            f.write(('  filter_wheel_positions: {}  # File containing resolver wheel positions for each filter/pupil\n'.format(input['filter_position_file'])))
            f.write(('  flux_cal: {} # File that lists flux conversion factor and pivot wavelength for each filter. Only '
                     'used when making direct image outputs to be fed into the grism disperser code.\n'.format(input['flux_cal_file'] )))
            f.write('  filter_throughput: {} #File containing filter throughput curve\n'.format(configfiles[instrument.lower()]['filter_throughput']))
            f.write('  ')
            f.write('\n')
            f.write('nonlin:\n')
            f.write('  limit: 60000.0                           # Upper singal limit to which nonlinearity is applied (ADU)\n')
            f.write('  accuracy: 0.000001                        # Non-linearity accuracy threshold\n')
            f.write('  maxiter: 10                              # Maximum number of iterations to use when applying non-linearity\n')
            f.write('  robberto:  False                         # Use Massimo Robberto type non-linearity coefficients\n')
            f.write('\n')
            f.write('cosmicRay:\n')
            cosmic_ray_path = os.path.join(datadir, instrument.lower(), 'cosmic_ray_library')
            f.write('  path: {}               # Path to CR library\n'.format(cosmic_ray_path))
            f.write('  library: {}    # Type of cosmic rayenvironment (SUNMAX, SUNMIN, FLARE)\n'.format(input['CosmicRayLibrary']))
            f.write('  scale: {}     # Cosmic ray scaling factor\n'.format(input['CosmicRayScale']))
            # temporary tweak here to make it work with NIRISS
            detector_label = input['detector']

            if instrument.lower() in ['nircam', 'wfsc']:
                # detector_label = input['detector']
                f.write('  suffix: IPC_NIRCam_{}    # Suffix of library file names\n'.format(
                    detector_label))
            elif instrument.lower() == 'niriss':
                f.write('  suffix: IPC_NIRISS_{}    # Suffix of library file names\n'.format(
                    detector_label))
            elif instrument.lower() == 'fgs':
                if detector_label == 'G1':
                    detector_string = 'GUIDER1'
                elif detector_label == 'G2':
                    detector_string = 'GUIDER2'
                f.write('  suffix: IPC_FGS_{}    # Suffix of library file names\n'.format(
                    detector_string))
            f.write('  seed: {}                 # Seed for random number generator\n'.format(np.random.randint(1, 2**32-2)))
            f.write('\n')
            f.write('simSignals:\n')
            if instrument.lower() in ['nircam', 'wfsc']:
                PointSourceCatalog = input['{}_ptsrc'.format(catkey)]
                GalaxyCatalog = input['{}_galcat'.format(catkey)]
                ExtendedCatalog = input['{}_ext'.format(catkey)]
                ExtendedScale = input['{}_extscl'.format(catkey)]
                ExtendedCenter = input['{}_extcent'.format(catkey)]
                MovingTargetList = input['{}_movptsrc'.format(catkey)]
                MovingTargetSersic = input['{}_movgal'.format(catkey)]
                MovingTargetExtended = input['{}_movext'.format(catkey)]
                MovingTargetConvolveExtended = input['{}_movconv'.format(catkey)]
                MovingTargetToTrack = input['{}_solarsys'.format(catkey)]
                ImagingTSOCatalog = input['{}_img_tso'.format(catkey)]
                GrismTSOCatalog = input['{}_grism_tso'.format(catkey)]
                BackgroundRate = input['{}_bkgd'.format(catkey)]
            elif instrument.lower() in ['niriss', 'fgs']:
                PointSourceCatalog = input['PointSourceCatalog']
                GalaxyCatalog = input['GalaxyCatalog']
                ExtendedCatalog = input['ExtendedCatalog']
                ExtendedScale = input['ExtendedScale']
                ExtendedCenter = input['ExtendedCenter']
                MovingTargetList = input['MovingTargetList']
                MovingTargetSersic = input['MovingTargetSersic']
                MovingTargetExtended = input['MovingTargetExtended']
                MovingTargetConvolveExtended = input['MovingTargetConvolveExtended']
                MovingTargetToTrack = input['MovingTargetToTrack']
                ImagingTSOCatalog = input['ImagingTSOCatalog']
                GrismTSOCatalog = input['GrismTSOCatalog']
                BackgroundRate = input['BackgroundRate']

            f.write(('  pointsource: {}   #File containing a list of point sources to add (x, y locations and magnitudes)\n'
                     .format(PointSourceCatalog)))
            f.write('  gridded_psf_library_row_padding: 4  # Number of outer rows and columns to avoid when evaluating library. RECOMMEND 4.\n')
            f.write('  psf_wing_threshold_file: {}   # File defining PSF sizes versus magnitude\n'.format(input['psf_wing_threshold_file']))
            f.write('  add_psf_wings: {}  # Whether or not to place the core of the psf from the gridded library into an image of the wings before adding.\n'.format(add_psf_wings))
            f.write('  psfpath: {}   #Path to PSF library\n'.format(input['psfpath']))
            f.write('  psfwfe: {}   #PSF WFE value (predicted or requirements)\n'.format(psfwfe))
            f.write('  psfwfegroup: {}      #WFE realization group (0 to 4)\n'.format(psfwfegroup))
            f.write(('  galaxyListFile: {}    #File containing a list of positions/ellipticities/magnitudes of galaxies '
                     'to simulate\n'.format(GalaxyCatalog)))
            f.write('  extended: {}          #Extended emission count rate image file name\n'.format(ExtendedCatalog))
            f.write('  extendedscale: {}                          #Scaling factor for extended emission image\n'.format(ExtendedScale))
            f.write(('  extendedCenter: {}                   #x, y pixel location at which to place the extended image '
                     'if it is smaller than the output array size\n'.format(ExtendedCenter)))
            f.write(('  PSFConvolveExtended: True #Convolve the extended image with the PSF before adding to the output '
                     'image (True or False)\n'))
            f.write(('  movingTargetList: {}          #Name of file containing a list of point source moving targets (e.g. '
                     'KBOs, asteroids) to add.\n'.format(MovingTargetList)))
            f.write(('  movingTargetSersic: {}  #ascii file containing a list of 2D sersic profiles to have moving through '
                     'the field\n'.format(MovingTargetSersic)))
            f.write(('  movingTargetExtended: {}      #ascii file containing a list of stamp images to add as moving targets '
                     '(planets, moons, etc)\n'.format(MovingTargetExtended)))
            f.write(('  movingTargetConvolveExtended: {}       #convolve the extended moving targets with PSF before adding.\n'
                     .format(MovingTargetConvolveExtended)))
            f.write(('  movingTargetToTrack: {} #File containing a single moving target which JWST will track during '
                     'observation (e.g. a planet, moon, KBO, asteroid)	This file will only be used if mode is set to '
                     '"moving_target" \n'.format(MovingTargetToTrack)))
            f.write(('  tso_imaging_catalog: {} #Catalog of (generally one) source for Imaging Time Series observation\n'.format(ImagingTSOCatalog)))
            f.write(('  tso_grism_catalog: {} #Catalog of (generally one) source for Grism Time Series observation\n'.format(GrismTSOCatalog)))
            f.write('  zodiacal:  None                          #Zodiacal light count rate image file \n')
            f.write('  zodiscale:  1.0                            #Zodi scaling factor\n')
            f.write('  scattered:  None                          #Scattered light count rate image file\n')
            f.write('  scatteredscale: 1.0                        #Scattered light scaling factor\n')
            f.write(('  bkgdrate: {}                         #Constant background count rate (ADU/sec/pixel in an undispersed image) or '
                     '"high","medium","low" similar to what is used in the ETC\n'.format(BackgroundRate)))
            f.write(('  poissonseed: {}                  #Random number generator seed for Poisson simulation)\n'
                     .format(np.random.randint(1, 2**32-2))))
            f.write('  photonyield: True                         #Apply photon yield in simulation\n')
            f.write('  pymethod: True                            #Use double Poisson simulation for photon yield\n')
            f.write('  expand_catalog_for_segments: {}                     # Expand catalog for 18 segments and use distinct PSFs\n'
                    .format(expand_catalog_for_segments))
            f.write('  use_dateobs_for_background: {}          # Use date_obs below to deternine background. If False, bkgdrate is used.\n'
                    .format(dateobs_for_background))
            f.write('  signal_low_limit_for_segmap: {}         # Lower signal limit to include pix in segmentation map. See below for units.\n'
                    .format(segmentation_threshold))
            f.write(('  signal_low_limit_for_segmap_units: {}  # Units of signal_low_limit_for_segmap. Can be: [ADU/sec, e/sec, MJy/sr, '
                     'ergs/cm2/a, ergs/cm2/hz]\n'.format(segmentation_threshold_units)))
            f.write('\n')
            f.write('Telescope:\n')
            f.write('  ra: {}                      # RA of simulated pointing\n'.format(input['ra_ref']))
            f.write('  dec: {}                    # Dec of simulated pointing\n'.format(input['dec_ref']))
            if 'pav3' in input.keys():
                pav3_value = input['pav3']
            else:
                pav3_value = input['PAV3']
            f.write('  rotation: {}                    # PA_V3 in degrees, i.e. the position angle of the V3 axis at V1 (V2=0, V3=0) measured from N to E.\n'.format(pav3_value))
            f.write('  tracking: {}   #Telescope tracking. Can be sidereal or non-sidereal\n'.format(input['Tracking']))
            f.write('\n')
            f.write('newRamp:\n')
            f.write('  dq_configfile: {}\n'.format(configfiles[instrument.lower()]['dq_init_config']))
            f.write('  sat_configfile: {}\n'.format(configfiles[instrument.lower()]['saturation_config']))
            f.write('  superbias_configfile: {}\n'.format(configfiles[instrument.lower()]['superbias_config']))
            f.write('  refpix_configfile: {}\n'.format(configfiles[instrument.lower()]['refpix_config']))
            f.write('  linear_configfile: {}\n'.format(configfiles[instrument.lower()]['linearity_config']))
            f.write('\n')
            f.write('Output:\n')
            # f.write('  use_stsci_output_name: {} # Output filename should follow STScI naming conventions (True/False)\n'.format(outtf))
            f.write('  directory: {}  # Output directory\n'.format(simdata_output_dir))
            f.write('  file: {}   # Output filename\n'.format(outfile))
            f.write(("  datatype: {} # Type of data to save. 'linear' for linearized ramp. 'raw' for raw ramp. 'linear, "
                     "raw' for both\n".format(datatype)))
            f.write('  format: DMS          # Output file format Options: DMS, SSR(not yet implemented)\n')
            f.write('  save_intermediates: False   # Save intermediate products separately (point source image, etc)\n')
            f.write('  grism_source_image: {}   # grism\n'.format(input['grism_source_image']))
            f.write('  unsigned: True   # Output unsigned integers? (0-65535 if true. -32768 to 32768 if false)\n')
            f.write('  dmsOrient: True    # Output in DMS orientation (vs. fitswriter orientation).\n')
            f.write('  program_number: {}    # Program Number\n'.format(input['ProposalID']))
            f.write('  title: {}   # Program title\n'.format(input['Title'].replace(':', ', ')))
            f.write('  PI_Name: {}  # Proposal PI Name\n'.format(input['PI_Name']))
            f.write('  Proposal_category: {}  # Proposal category\n'.format(input['Proposal_category']))
            f.write('  Science_category: {}  # Science category\n'.format(input['Science_category']))
            f.write('  target_name: {}  # Name of target\n'.format(input['TargetID']))

            # For now, skip populating the target RA and Dec in WFSC data.
            # The read_xxxxx funtions for these observation types will have
            # to be updated to make use of the proposal_parameter_dictionary
            if np.isreal(input['TargetRA']):
                input['TargetRA'] = str(input['TargetRA'])
                input['TargetDec'] = str(input['TargetDec'])
            if input['TargetRA'] != '0':
                ra_degrees, dec_degrees = utils.parse_RA_Dec(input['TargetRA'], input['TargetDec'])
            else:
                ra_degrees = 0.
                dec_degrees = 0.

            f.write('  target_ra: {}  # RA of the target, from APT file.\n'.format(ra_degrees))
            f.write('  target_dec: {}  # Dec of the target, from APT file.\n'.format(dec_degrees))
            f.write("  observation_number: '{}'    # Observation Number\n".format(input['obs_num']))
            f.write('  observation_label: {}    # User-generated observation Label\n'.format(input['obs_label'].strip()))
            f.write("  visit_number: '{}'    # Visit Number\n".format(input['visit_num']))
            f.write("  visit_group: '{}'    # Visit Group\n".format(input['visit_group']))
            f.write("  visit_id: '{}'    # Visit ID\n".format(input['visit_id']))
            f.write("  sequence_id: '{}'    # Sequence ID\n".format(input['sequence_id']))
            f.write("  activity_id: '{}'    # Activity ID. Increment with each exposure.\n".format(input['act_id']))
            f.write("  exposure_number: '{}'    # Exposure Number\n".format(input['exposure']))
            f.write("  obs_id: '{}'   # Observation ID number\n".format(input['observation_id']))
            f.write("  date_obs: '{}'  # Date of observation\n".format(input['date_obs']))
            f.write("  time_obs: '{}'  # Time of observation\n".format(input['time_obs']))
            # f.write("  obs_template: '{}'  # Observation template\n".format(input['obs_template']))
            f.write("  primary_dither_type: {}  # Primary dither pattern name\n".format(input['PrimaryDitherType']))
            f.write("  total_primary_dither_positions: {}  # Total number of primary dither positions\n".format(input['PrimaryDithers']))
            f.write("  primary_dither_position: {}  # Primary dither position number\n".format(np.int(input['primary_dither_num'])))
            f.write("  subpix_dither_type: {}  # Subpixel dither pattern name\n".format(input['SubpixelDitherType']))
            # For WFSS we need to strip out the '-Points' from
            # the number of subpixel positions entry
            dash = -1
            if isinstance(input['SubpixelPositions'], str):
                dash = input['SubpixelPositions'].find('-')
            if (dash == -1):
                val = input['SubpixelPositions']
            else:
                val = input['SubpixelPositions'][0:dash]
            if val == 'None':
                val = 1
            # try:
            #     dash = input['SubpixelPositions'].find('-')
            #     val = input['SubpixelPositions'][0:dash]
            # except:
            #     val = input['SubpixelPositions']
            f.write("  total_subpix_dither_positions: {}  # Total number of subpixel dither positions\n".format(val))
            f.write("  subpix_dither_position: {}  # Subpixel dither position number\n".format(np.int(input['subpix_dither_num'])))
            f.write("  xoffset: {}  # Dither pointing offset in x (arcsec)\n".format(input['idlx']))
            f.write("  yoffset: {}  # Dither pointing offset in y (arcsec)\n".format(input['idly']))

        return yamlout

    def add_options(xml_file = 'imaging_example_data/example_imaging_program.xml',
        pointing_file = 'imaging_example_data/example_imaging_program.pointing',
        cat_dict = {'GOODS-S-FIELD': {'point_source':'imaging_example_data/ptsrc_catalog.cat'}},
        reffile_defaults = 'crds',
        cosmic_rays = {'library': 'SUNMAX', 'scale': 1.0},
        background = 'medium',
        pav3 = 12.5,
        dates = '2022-10-31',
        output_dir = './imaging_example_data/',
        simulation_dir = './imaging_example_data/',
        datatype = 'linear, raw', parser=None, usage=None):
        if parser is None:
            parser = argparse.ArgumentParser(usage=usage, description='Simulate JWST ramp')
        parser.add_argument("--input_xml", help='XML file from APT describing the observations.')
        parser.add_argument("--pointing_file", help='Pointing file from APT describing observations.')
        parser.add_argument("--datatype", help='Type of data to save. Can be "linear", "raw" or "linear, raw"', default="linear")
        parser.add_argument("--output_dir", help='Directory into which the yaml files are output', default='./')
        parser.add_argument("--table_file", help='Ascii table containing observation info. Use this or xml + pointing files.', default=None)
        parser.add_argument("--use_nonstsci_names", help="Use STScI naming convention for output files", action='store_true')
        parser.add_argument("--subarray_def_file", help="Ascii file containing subarray definitions", default='config')
        parser.add_argument("--readpatt_def_file", help='Ascii file containing readout pattern definitions', default='config')
        parser.add_argument("--crosstalk", help="Crosstalk coefficient file", default='config')
        parser.add_argument("--filtpupil_pairs", help="List of paired filter/pupil elements", default='config')
        parser.add_argument("--fluxcal", help="File with zeropoints per filter", default='config')
        parser.add_argument("--dq_init_config", help="DQ Initialization config file", default='config')
        parser.add_argument("--saturation_config", help="Saturation config file", default='config')
        parser.add_argument("--superbias_config", help="Superbias subtraction config file", default='config')
        parser.add_argument("--refpix_config", help="Refpix subtraction config file", default='config')
        parser.add_argument("--linearity_config", help="Linearity config file", default='config')
        parser.add_argument("--observation_list_file", help="Table file containing epoch start times, telescope roll angles, catalogs for each observation", default=None)
        parser.add_argument("--use_JWST_pipeline", help='True/False', action='store_true')
        parser.add_argument("--use_linearized_darks", help='True/False', action='store_true')
        parser.add_argument("--simdata_output_dir", help='Output directory for simulated exposure files', default='./')
        parser.add_argument("--psfpath", help='Directory containing PSF library',
                            default=os.path.join(datadir, 'gridded_psf_library'))
        parser.add_argument("--psfbasename", help="Basename of the files in the PSF library", default='nircam')
        parser.add_argument("--psfpixfrac", help="Subpixel centering resolution of files in PSF library", default=0.25)
        parser.add_argument("--psfwfe", help="Wavefront error value to use for PSFs", default='predicted')
        parser.add_argument("--psfwfegroup", help="Realization index number for PSF files", default=0)
        parser.add_argument("--resets_bet_ints", help="Number of detector resets between integrations", default=1)
        parser.add_argument("--tracking", help="Type of telescope tracking: 'sidereal' or 'non-sidereal'", default='sidereal')

        return parser


def _gtvt_v3pa_on_date(ra, dec, date=None, return_range=False):
    """Call JWST GTVT to retrieve nominal observatory position angle (V3 PA) for given coordinates and date

    This is a lower-level util function; most MIRAGE users should typically use default_obs_v3pa_on_date or
    all_obs_v3pa_on_date instead.

    Parameters
    ----------
    ra : str
        RA in sexagesimal format e.g. '00:00:00.0'

    dec : str
        Dec in sexagesimal format e.g. '00:00:00.0'

    date : string
        Desired date of observation, in YYYY-MM-DD format. If not supplied, the current date will be used.

    return_range : bool
        Return tuple including the range accessible via observatory roll, i.e. (v3pa, min_v3pa, max_v3pa)

    Returns
    -------
    pa_v3 : float
        V3PA in degrees
    """
    import jwst_gtvt.find_tgt_info

    if date is None:
        start_date_obj = datetime.date.today()
        start_date = start_date_obj.isoformat()
    else:
        start_date_obj = datetime.datetime.strptime(date, '%Y-%m-%d')
        start_date = start_date_obj.isoformat().split('T')[0]

    # note, get_table requires distinct start and end dates, different by at least 1 day
    end_date_obj = start_date_obj + datetime.timedelta(days=1)
    end_date = end_date_obj.isoformat().split('T')[0]

    tbl = jwst_gtvt.find_tgt_info.get_table(ra=ra, dec=dec, instrument='NIRCam',
                                            start_date=start_date, end_date=end_date,
                                            verbose=False)
    row = tbl[0]

    if return_range:
        return row['V3PA'], row['V3PA min'], row['V3PA max']
    return row['V3PA']


def default_obs_v3pa_on_date(pointing_filename, obs_num, date=None, verbose=False, pointing_table=None):
    """Find the nominal/default V3PA for an observation on a given date.

    Note, this relies on the JWST Generalized Target Visibility Tool (GTVT), and as such it
    just considers the basic spherical geometry of the sky. It does *NOT* take into account
    any special requirements defined in APT.

    Parameters
    ----------
    pointing_filename : string
        Pointing file filename exported by APT
    obs_num : int
        Observation number.
    date : str or None
        Desired date of observation, in YYYY-MM-DD format. If not supplied, the current date will be used.
    pointing_table : dict, optional
        Dictionary of info read from pointing filename; alternate input for this in case it's already been read from disk

    Returns
    -------
    pa : float
        V3PA in decimal degrees. Provide this to the `roll_angle` argument to mirage.yaml_generator()
    """

    if pointing_table is None:
        pointing_table = apt_inputs.AptInput().get_pointing_info(pointing_filename, 0)
    for i in range(len(pointing_table['obs_num'])):
        if pointing_table['obs_num'][i] == f"{obs_num:03d}":
            ra_deg, dec_deg = pointing_table['ra'][i], pointing_table['dec'][i]
            if verbose:
                logger.info(f"Pointing table row {i} is for obs {obs_num}")
                logger.info(f" Coords from APT pointing file: {ra_deg} {dec_deg} deg")
            break
    else:
        raise RuntimeError(f"Could not find any info for an observation number {obs_num} in the pointing table.")

    result = _gtvt_v3pa_on_date(ra_deg, dec_deg, date=date)
    if np.isnan(result):
        raise RuntimeError("Obs {obs_num} is not observable on date {date}. Target not in the field of regard.")
    return result


def all_obs_v3pa_on_date(pointing_filename, date=None, verbose=False):
    """Find the nominal/default V3PA for all observations in a program.

    Parameters
    ----------
    pointing_filename : string
        Pointing file filename exported by APT
    date : str or None
        Desired date of observation, in YYYY-MM-DD format. If not supplied, the current date will be used.


    Returns
    -------
    pas : dict
        Dict of V3PAs, in the format for passing to the `roll_angle` argument to mirage.yaml_generator()

    """
    results = {}
    pointing_table = apt_inputs.AptInput().get_pointing_info(pointing_filename, 0)
    obsnums = sorted(list(set(pointing_table['obs_num'])))
    for obs_num in obsnums:
        results[obs_num] = default_obs_v3pa_on_date(pointing_filename, int(obs_num), date=date, verbose=verbose,
                                                    pointing_table=pointing_table)
    return results


if __name__ == '__main__':

    usagestring = 'USAGE: yaml_generator.py NIRCam_obs.xml NIRCam_obs.pointing'

    input = SimInput()
    parser = input.add_options(usage=usagestring)
    args = parser.parse_args(namespace=input)
    input.reffile_setup()
    input.create_inputs()
""""""
