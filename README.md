# ee547-hw1-linlinhua-

- Lin-Lin Hua
- linlinhu@usc.edu
- (Any external libraries used beyond those specified): No
- Instructions to run each problem if they differ from the assignment specification:
  - Problem 1: 
    1. python fetch_and_process.py urls.txt out_dir
  - Problem 2:
    1. ./build.sh
    2. ./run.sh "cat:cs.LG" 10 output/
  - Problem 3:
    ##create dirs and enter urls.txt
    1. mkdir -p shared/{input,raw,processed,status,analysis}
    2. cp test_urls.txt shared/input/urls.txt
    
    ##remove the old containers
    4. (Optional) docker compose down #remove the old containers
    5. docker-compose build



