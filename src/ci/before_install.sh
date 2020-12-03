sudo apt-get update
sudo apt-get install nodejs -q -y
# Elasticsearch 7.x installation
sudo apt purge elasticsearch
sudo apt-get install apt-transport-https
echo "deb https://artifacts.elastic.co/packages/7.x/apt stable main" | sudo tee /etc/apt/sources.list.d/elastic-7.x.list
sudo apt-get update && sudo apt-get install elasticsearch=7.4.2 -q -y --allow-unauthenticated
sudo sudo systemctl enable elasticsearch
sudo systemctl start elasticsearch
sudo systemctl status elasticsearch
