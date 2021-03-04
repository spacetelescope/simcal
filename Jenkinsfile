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
        //      sh "mvn install"
                sh("mkdir -p tmp")
        //      sh("curl https://repo.anaconda.com/miniconda/Miniconda-2.0.3-MacOSX-x86.sh -o installer.sh")
        //      sh("bash installer.sh -b -p ${WORKSPACE}/miniconda3")
        //      sh("curl -LO https://raw.githubusercontent.com/astroconda/docker-buildsys/master/with_env")



                sh("wget https://repo.anaconda.com/miniconda/Miniconda3-latest-MacOSX-x86_64.sh -O ~/miniconda.sh")
        //      sh("bash ~/miniconda.sh -b -p $HOME/miniconda")


                sh("bash ~/miniconda.sh -b -h $HOME/miniconda")


                sh("chmod +x with_env")
                sh("conda env create -n ${env_name} -f environment.yml")
                }
                }

               stage('Install') {
               steps {
               sh("./with_env -n ${env_name} pip install -f .") }}

               }
               }
