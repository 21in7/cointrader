pipeline {
    agent any

    environment {
        REGISTRY      = '10.1.10.28:3000'
        IMAGE_NAME    = 'gihyeon/cointrader'
        IMAGE_TAG     = "${env.BUILD_NUMBER}"
        FULL_IMAGE    = "${REGISTRY}/${IMAGE_NAME}:${IMAGE_TAG}"
        LATEST_IMAGE  = "${REGISTRY}/${IMAGE_NAME}:latest"
    }

    stages {
        stage('Git Clone from Gitea') {
            steps {
                git branch: 'main',
                    credentialsId: 'gitea-cred',
                    url: 'http://10.1.10.28:3000/gihyeon/cointrader.git'
            }
        }

        stage('Build Docker Image') {
            steps {
                sh "docker build -t ${FULL_IMAGE} -t ${LATEST_IMAGE} ."
            }
        }

        stage('Push to Gitea Registry') {
            steps {
                withCredentials([usernamePassword(credentialsId: 'gitea-registry-cred', passwordVariable: 'GITEA_TOKEN', usernameVariable: 'GITEA_USER')]) {
                    sh "echo \$GITEA_TOKEN | docker login ${REGISTRY} -u \$GITEA_USER --password-stdin"
                    sh "docker push ${FULL_IMAGE}"
                    sh "docker push ${LATEST_IMAGE}"
                }
            }
        }

        stage('Deploy to Prod LXC') {
            steps {
                sh 'ssh root@10.1.10.24 "mkdir -p /root/cointrader"'
                sh 'scp docker-compose.yml root@10.1.10.24:/root/cointrader/'
                sh '''
                    ssh root@10.1.10.24 "cd /root/cointrader/ && \
                    docker compose down && \
                    docker compose pull && \
                    docker compose up -d"
                '''
            }
        }

        stage('Cleanup') {
            steps {
                sh "docker rmi ${FULL_IMAGE} || true"
                sh "docker rmi ${LATEST_IMAGE} || true"
            }
        }
    }

    post {
        success {
            echo "Build #${env.BUILD_NUMBER} 성공: ${FULL_IMAGE} → 운영 LXC(10.1.10.24) 배포 완료"
        }
        failure {
            echo "Build #${env.BUILD_NUMBER} 실패"
        }
    }
}
