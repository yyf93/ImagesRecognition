#crontab -e
#0 */3 * * * source /home/ubuntu/.bashrc && sh /home/ubuntu/ImagesRecognition/clean_jpg.sh
#0 */3 * * * source /home/ubuntu/.bashrc && sh /home/ubuntu/ImagesRecognition/restart_server.sh

ps -ef | grep "server.py" | awk '{print $2}' | xargs kill -9
python3 /home/ubuntu/ImagesRecognition/face.py &
