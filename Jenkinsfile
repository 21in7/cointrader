pipeline {
    agent any

    environment {
        REGISTRY      = 'git.gihyeon.com'
        IMAGE_NAME    = 'gihyeon/cointrader'
        IMAGE_TAG     = "${env.BUILD_NUMBER}"
        FULL_IMAGE    = "${REGISTRY}/${IMAGE_NAME}:${IMAGE_TAG}"
        LATEST_IMAGE  = "${REGISTRY}/${IMAGE_NAME}:latest"

        DASH_API_IMAGE  = "${REGISTRY}/gihyeon/cointrader-dashboard-api"
        DASH_UI_IMAGE   = "${REGISTRY}/gihyeon/cointrader-dashboard-ui"

        DISCORD_WEBHOOK = credentials('discord-webhook')
    }

    stages {
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
                    url: 'https://git.gihyeon.com/gihyeon/cointrader.git'
            }
        }

        stage('Detect Changes') {
            steps {
                script {
                    // 이전 성공 빌드 커밋과 비교 (없으면 HEAD~5 fallback)
                    def baseCommit = env.GIT_PREVIOUS_SUCCESSFUL_COMMIT ?: sh(script: 'git rev-parse HEAD~5 2>/dev/null || echo ""', returnStdout: true).trim()
                    def diffCmd = baseCommit ? "git diff --name-only ${baseCommit}..HEAD" : 'git diff --name-only HEAD~1'
                    def changes = sh(script: "${diffCmd} || echo \"ALL\"", returnStdout: true).trim()
                    echo "Base commit: ${baseCommit ?: 'HEAD~1 (fallback)'}"
                    echo "Changed files:\n${changes}"

                    if (changes == 'ALL') {
                        // 첫 빌드이거나 diff 실패 시 전체 빌드
                        env.BOT_CHANGED = 'true'
                        env.DASH_API_CHANGED = 'true'
                        env.DASH_UI_CHANGED = 'true'
                    } else {
                        env.BOT_CHANGED = (changes =~ /(?m)^(src\/|main\.py|requirements\.txt|Dockerfile)/).find() ? 'true' : 'false'
                        env.DASH_API_CHANGED = (changes =~ /(?m)^dashboard\/api\//).find() ? 'true' : 'false'
                        env.DASH_UI_CHANGED = (changes =~ /(?m)^dashboard\/ui\//).find() ? 'true' : 'false'
                    }

                    // docker-compose.yml 변경 시에도 배포 필요
                    if (changes.contains('docker-compose.yml') || changes.contains('Jenkinsfile')) {
                        env.COMPOSE_CHANGED = 'true'
                    } else {
                        env.COMPOSE_CHANGED = 'false'
                    }

                    echo "BOT_CHANGED=${env.BOT_CHANGED}, DASH_API_CHANGED=${env.DASH_API_CHANGED}, DASH_UI_CHANGED=${env.DASH_UI_CHANGED}, COMPOSE_CHANGED=${env.COMPOSE_CHANGED}"
                }
            }
        }

        stage('Build Docker Images') {
            parallel {
                stage('Bot') {
                    when { expression { env.BOT_CHANGED == 'true' } }
                    steps {
                        sh "docker build -t ${FULL_IMAGE} -t ${LATEST_IMAGE} ."
                    }
                }
                stage('Dashboard API') {
                    when { expression { env.DASH_API_CHANGED == 'true' } }
                    steps {
                        sh "docker build -t ${DASH_API_IMAGE}:${IMAGE_TAG} -t ${DASH_API_IMAGE}:latest ./dashboard/api"
                    }
                }
                stage('Dashboard UI') {
                    when { expression { env.DASH_UI_CHANGED == 'true' } }
                    steps {
                        sh "docker build -t ${DASH_UI_IMAGE}:${IMAGE_TAG} -t ${DASH_UI_IMAGE}:latest ./dashboard/ui"
                    }
                }
            }
        }

        stage('Push to Gitea Registry') {
            steps {
                withCredentials([usernamePassword(credentialsId: 'gitea-registry-cred', passwordVariable: 'GITEA_TOKEN', usernameVariable: 'GITEA_USER')]) {
                    sh "echo \$GITEA_TOKEN | docker login ${REGISTRY} -u \$GITEA_USER --password-stdin"
                    script {
                        if (env.BOT_CHANGED == 'true') {
                            sh "docker push ${FULL_IMAGE}"
                            sh "docker push ${LATEST_IMAGE}"
                        }
                        if (env.DASH_API_CHANGED == 'true') {
                            sh "docker push ${DASH_API_IMAGE}:${IMAGE_TAG}"
                            sh "docker push ${DASH_API_IMAGE}:latest"
                        }
                        if (env.DASH_UI_CHANGED == 'true') {
                            sh "docker push ${DASH_UI_IMAGE}:${IMAGE_TAG}"
                            sh "docker push ${DASH_UI_IMAGE}:latest"
                        }
                    }
                }
            }
        }

        stage('Deploy to Prod LXC') {
            steps {
                script {
                    // docker-compose.yml이 변경되었으면 항상 전송
                    if (env.COMPOSE_CHANGED == 'true') {
                        sh 'ssh root@10.1.10.24 "mkdir -p /root/cointrader"'
                        sh 'scp docker-compose.yml root@10.1.10.24:/root/cointrader/'
                    }

                    // 변경된 서비스만 pull & recreate (나머지는 중단 없음)
                    def services = []
                    if (env.BOT_CHANGED == 'true') services.add('cointrader')
                    if (env.DASH_API_CHANGED == 'true') services.add('dashboard-api')
                    if (env.DASH_UI_CHANGED == 'true') services.add('dashboard-ui')

                    if (env.COMPOSE_CHANGED == 'true' && services.isEmpty()) {
                        // compose만 변경된 경우 전체 재시작
                        sh 'ssh root@10.1.10.24 "cd /root/cointrader/ && docker compose up -d"'
                    } else if (!services.isEmpty()) {
                        def svcList = services.join(' ')
                        sh "ssh root@10.1.10.24 \"cd /root/cointrader/ && docker compose pull ${svcList} && docker compose up -d ${svcList}\""
                    }
                }
            }
        }

        stage('Cleanup') {
            steps {
                script {
                    if (env.BOT_CHANGED == 'true') {
                        sh "docker rmi ${FULL_IMAGE} || true"
                        sh "docker rmi ${LATEST_IMAGE} || true"
                    }
                    if (env.DASH_API_CHANGED == 'true') {
                        sh "docker rmi ${DASH_API_IMAGE}:${IMAGE_TAG} || true"
                        sh "docker rmi ${DASH_API_IMAGE}:latest || true"
                    }
                    if (env.DASH_UI_CHANGED == 'true') {
                        sh "docker rmi ${DASH_UI_IMAGE}:${IMAGE_TAG} || true"
                        sh "docker rmi ${DASH_UI_IMAGE}:latest || true"
                    }
                }
            }
        }
    }

    post {
        success {
            echo "Build #${env.BUILD_NUMBER} 성공"
            sh """
            curl -H "Content-Type: application/json" \
                 -X POST \
                 -d '{"content": "✅ **[배포 성공]** `cointrader` (Build #${env.BUILD_NUMBER}) 운영 서버(10.1.10.24) 배포 완료!\\n- 🤖 봇: ${env.BOT_CHANGED}\\n- 📊 API: ${env.DASH_API_CHANGED}\\n- 🖥️ UI: ${env.DASH_UI_CHANGED}"}' \
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
