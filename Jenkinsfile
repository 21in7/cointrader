pipeline {
    agent any

    environment {
        REGISTRY      = '10.1.10.28:3000'
        IMAGE_NAME    = 'gihyeon/cointrader'
        IMAGE_TAG     = "${env.BUILD_NUMBER}"
        FULL_IMAGE    = "${REGISTRY}/${IMAGE_NAME}:${IMAGE_TAG}"
        LATEST_IMAGE  = "${REGISTRY}/${IMAGE_NAME}:latest"
        
        // 젠킨스 자격 증명에 저장해둔 디스코드 웹훅 주소를 불러옵니다.
        DISCORD_WEBHOOK = credentials('discord-webhook')
    }

    stages {
        // 빌드가 시작되자마자 알림을 보냅니다.
        stage('Notify Build Start') {
            steps {
                sh """
                curl -H "Content-Type: application/json" \
                     -X POST \
                     -d '{"content": "🚀 **[빌드 시작]** `cointrader` (Build #${env.BUILD_NUMBER}) 배포 파이프라인 가동"}' \
                     ${DISCORD_WEBHOOK}
                """
            }
        }

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

    // 파이프라인 결과에 따른 디스코드 알림
    post {
        success {
            echo "Build #${env.BUILD_NUMBER} 성공: ${FULL_IMAGE} → 운영 LXC(10.1.10.24) 배포 완료"
            sh """
            curl -H "Content-Type: application/json" \
                 -X POST \
                 -d '{"content": "✅ **[배포 성공]** `cointrader` (Build #${env.BUILD_NUMBER}) 운영 서버(10.1.10.24) 배포 완료!\\n- 📦 이미지: `${FULL_IMAGE}`"}' \
                 ${DISCORD_WEBHOOK}
            """
        }
        failure {
            echo "Build #${env.BUILD_NUMBER} 실패"
            sh """
            curl -H "Content-Type: application/json" \
                 -X POST \
                 -d '{"content": "❌ **[배포 실패]** `cointrader` (Build #${env.BUILD_NUMBER}) 파이프라인 에러 발생. 젠킨스 로그를 확인해 주세요!"}' \
                 ${DISCORD_WEBHOOK}
            """
        }
    }
}