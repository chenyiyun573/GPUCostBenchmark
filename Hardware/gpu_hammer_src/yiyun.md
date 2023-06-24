In kernel_fp.cu, about why we add judgement statement to prevent optimization:

 this if statement and the following code block serve a purpose: they help prevent compiler optimizations that could potentially remove the FMA operations if the compiler determines that their results are not used. By having this if statement and updating the values of sfma_out and dfma_out, the developer ensures that the FMA operations in the loop are not removed by the compiler, as their results could potentially affect the final values of sfma_out and dfma_out.

