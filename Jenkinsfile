pipeline {
  agent { label 'RHEL7' }

  environment {
    env_name = "simcal"

    HOME="${WORKSPACE}"
    MIRAGE_DATA="/ifs/jwst/wit/mirage_data/"
    TEST_BIGDATA="https://bytesalad.stsci.edu/artifactory"
    CRDS_SERVER_URL = "https://jwst-crds.stsci.edu"
    CRDS_PATH = "/tmp/crds_cache"
    PATH ="${WORKSPACE}/miniconda3/bin:${PATH}"
    TMPDIR="${WORKSPACE}/tmp"
    XDG_CACHE_HOME="${WORKSPACE}/tmp/.cache"
  }

  stages {
    stage('Setup') {
      steps {
        deleteDir()
        checkout scm
        sh("mkdir -p tmp")
        sh("curl https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-x86_64.sh -o installer.sh")
        sh("bash installer.sh -b -p ${WORKSPACE}/miniconda3")
        sh("curl -LO https://raw.githubusercontent.com/astroconda/docker-buildsys/master/with_env")
        sh("chmod +x with_env")
        sh("conda env create -n ${env_name} -f environment.yml")
      }
    }

    stage('Install') {
      steps {
        sh("./with_env -n ${env_name} pip install -e .")
        }
    }

    

//    stage('Convert/Check') {
//      steps {
//        sh("./with_env -n ${env_name} python convert.py --notebook-path jwst_validation_notebooks --report report.txt")
//        sh("./with_env -n ${env_name} python prepend_date.py --reportfile report.xml")
//        sh("""
//              cd jwst_validation_notebooks
//              ../with_env -n ${env_name} python -m 'nbpages.check_nbs'
//           """)
//      }
//    }


//        } // end of script
//      } // end of deploy steps
//    } // end of deploy stage
//   } // end of stages


  post {
        cleanup {
            deleteDir()
        } //end of cleanup
  } //end of post
} // end of pipeline
