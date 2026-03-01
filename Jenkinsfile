pipeline {
    agent any

    environment {
        REGISTRY      = '10.1.10.28:3000'
        IMAGE_NAME    = 'gihyeon/cointrader'
        IMAGE_TAG     = "${env.BUILD_NUMBER}"
        FULL_IMAGE    = "${REGISTRY}/${IMAGE_NAME}:${IMAGE_TAG}"
        LATEST_IMAGE  = "${REGISTRY}/${IMAGE_NAME}:latest"
        GITEA_CREDS   = credentials('gitea-registry-credentials')
    }

    stages {
        stage('Checkout') {
            steps {
                checkout scm
            }
        }

        stage('Build Image') {
            steps {
                sh "docker build -t ${FULL_IMAGE} -t ${LATEST_IMAGE} ."
            }
        }

        stage('Push to Gitea Registry') {
            steps {
                sh """
                    echo ${GITEA_CREDS_PSW} | docker login ${REGISTRY} -u ${GITEA_CREDS_USR} --password-stdin
                    docker push ${FULL_IMAGE}
                    docker push ${LATEST_IMAGE}
                """
            }
        }

        stage('Cleanup') {
            steps {
                sh """
                    docker rmi ${FULL_IMAGE} || true
                    docker rmi ${LATEST_IMAGE} || true
                """
            }
        }
    }

    post {
        success {
            echo "Build #${env.BUILD_NUMBER} pushed: ${FULL_IMAGE}"
        }
        failure {
            echo "Build #${env.BUILD_NUMBER} FAILED"
        }
    }
}
