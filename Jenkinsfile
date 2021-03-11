pipeline {
    agent any
      environment {

       HOME="${WORKSPACE}"
       MIRAGE_DATA="/ifs/jwst/wit/mirage_data/"
       TEST_BIGDATA="https://bytesalad.stsci.edu/artifactory"
       CRDS_SERVER_URL = "https://jwst-crds.stsci.edu"
       CRDS_PATH = "/tmp/crds_cache"
       PATH ="${WORKSPACE}/miniconda3/bin:${PATH}"
       TMPDIR="${WORKSPACE}/tmp"
       XDG_CACHE_HOME="${WORKSPACE}/tmp/.cache"

   }
      stages{
        stage('Setup') {
            steps {
                deleteDir()
                checkout scm
                sh("mkdir -p tmp")
                sh("curl https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-x86_64.sh -o miniconda.sh")
                sh("bash miniconda.sh -b -p ${WORKSPACE}/miniconda3")
                sh("chmod +x with_env")

                }
                }

               stage('Install') {
               steps {
               sh("./with_env -n ${env_name} pip install -f .") }}

               }
               }
