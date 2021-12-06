# Requirements
The code was tested on MacBook Air M1.
<br>
<li>
    Opencv
</li>
<li>
    Numpy
 </li>
 
 # Running the program
 Download <a href="https://drive.google.com/file/d/1GWFq4kNcGXiywyELpRgQrpc4Ahd6XUsO/view?usp=sharing" > this </a> video and save it in the root directory. Make sure that the file name is right_sample2.mov 
 <br>
 All the files should be in same directory
 <br>
 mulprocess.py utilizes Python's multiprocess threadpool module
 <br>
 multhread.py utilizes Python's threading module by dividing the input in chunks
 <br>
 Execute like a simple Python script
 <br>
 ```time python3 mulprocess.py ```
 <br>
 The output is stored in labels.txt
 <br>
 labels.txt should be empty before running the script
 
 
 # Key Points
 It might not give same time as reported. We tested the code on another Macbook Air M1 and numbers were quite different but serial code always took longer than multithreaded and multiprocessed code.

 
