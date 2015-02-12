#include "fann.h"
#include <stdio.h>
#include <OpenCL/cl.h>

int main()
{
  
   
    fann_type *calc_out; 
    fann_type *calc_out2;
    unsigned int i;
    int incorrect,ret = 0;
    int counter = 0;
    
    FILE *fp;
    FILE *fp2;

    char *features[] = {
    "left_eye_center_x","left_eye_center_y","right_eye_center_x","right_eye_center_y",
    "left_eye_inner_corner_x","left_eye_inner_corner_y","left_eye_outer_corner_x",
    "left_eye_outer_corner_y","right_eye_inner_corner_x","right_eye_inner_corner_y",
    "right_eye_outer_corner_x","right_eye_outer_corner_y","left_eyebrow_inner_end_x",  
    "left_eyebrow_inner_end_y","left_eyebrow_outer_end_x","left_eyebrow_outer_end_y",
    "right_eyebrow_inner_end_x", "right_eyebrow_inner_end_y","right_eyebrow_outer_end_x", 
    "right_eyebrow_outer_end_y","nose_tip_x","nose_tip_y","mouth_left_corner_x","mouth_left_corner_y",
    "mouth_right_corner_x","mouth_right_corner_y","mouth_center_top_lip_x","mouth_center_top_lip_y",
    "mouth_center_bottom_lip_x","mouth_center_bottom_lip_y"
    
    };
    

    printf("Testing network..\n");
    
    fp = fopen("result.txt", "w");
	
	
	fprintf(fp, "RowId,ImageId,FeatureName,Location\n");

	struct fann *ann = fann_create_from_file("facial.net"); 
	struct fann *ann2 = fann_create_from_file("facial2.net"); 
	   
    struct fann_train_data *data = fann_read_train_from_file("ftest.txt");
    
    
    
    for(i = 0; i < fann_length_train_data(data); i++) {
        
       
        calc_out = fann_run(ann, data->input[i] );
        calc_out2 = fann_run(ann2, data->input[i]);
        
    	
        
        for (int n=0; n<30; n++) {
            counter++;
             unsigned int output;
            
            output = (calc_out[n]*6 + calc_out2[n]*6) / 2;
	        //printf (" %i#  %.2f/%.2f > %.2f \n", i+1 , calc_out[n]*6, calc_out2[n]*6, output );

            
            fprintf(fp, "%i,%i,%s,%i\n",counter,i+1,features[n], (int)calc_out[n]);
                   
        }
        
        counter++; 
        
       
       
    }
    
    fclose(fp);
    printf("Mean Square Error: %f\n", fann_get_MSE(ann));
    

	fann_destroy_train(data);
	fann_destroy(ann);


	
    return 0;
}