# Define flags
while getopts ":hrc" opt; do
  case ${opt} in
    h )
      echo "Usage:"
      echo "  start_main.sh -h         Display this help message"
      echo "  start_main.sh -r         Rebuild Docker image then run"
      exit 1
      ;;
    r )
      echo "Rebuilding Docker image"
      docker build -t lmpc_docker:latest ./docker # Build image
      untagged_imgs=$(docker images | grep "^<none>" | awk '{print $3}') # Get list of untagged images
      if [[ !  -z  ${untagged_imgs}  ]]
      then
        echo "Deleting untagged images"
        docker rmi ${untagged_imgs} # Remove untagged images
      fi
      ;;
    \? )
      echo "Invalid option"
      exit 1
      ;;
  esac
done

xhost +

DATE=$( date +%N )
BASE_DIR=$( pwd )

# Start and run the container
docker run --name LMPC_${DATE} -it --rm \
  -e DISPLAY=${DISPLAY} \
  --mount type=bind,src=/tmp/.X11-unix,dst=/tmp/.X11-unix \
  --mount type=bind,src=${BASE_DIR},dst=/home/LMPC \
  -p 8888:8888 \
  lmpc_docker:latest
