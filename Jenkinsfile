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




    stages {
        stage('Checkout Code') {
            steps {
                // Get some code from a GitHub repository
                git 'https://github.com/spacetelescope/simcal.git'



                stage('Run Unit Test Cases') {
                    steps {
                        bat "mvn clean test"

                    }
                stage('Build Code') {
                    steps {
                        junit '**/target/surefire-reports/TEST-*.xml'
                    }
                }

                stage('Build Code'){
                    steps {
                        bat "mvn package -DskipTests=true"
                    }
                }
                stage('Archive Results')
                {
                    steps {
                        archiveArtifacts 'target/*.war'
                    }
        }
     }
  }
}
}
}
