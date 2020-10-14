# remove-logo
A barebone implementation of a logo removal algorithm,only using (core) opencv

https://www.youtube.com/playlist?list=PLf37j7i5CN0Ex0i8pChLNcWQkfnY-PJyT


## Prerequisites
- C(>= ++11 required)
- opencv (>= 4.0.0)

## Getting started
- Prepare the video and the reference image.
- Set the ROI boundaries
- Compile and run it:
```bash
make main
./main
```

### Algorithm assumptions
- The camera is fixed
- a cleared reference image is provided

### Improvements
- Replace classical approaches with ML based solutions. (for example for text detection or for humans segmentation)
- Implement fault handlings
- Speed up/parallelize the code
- lots of others!

