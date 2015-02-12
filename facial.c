#include "fann.h"
#include <OpenCL/cl.h>

int main()
{
    const unsigned int max_epochs = 1500;
    const unsigned int epochs_between_reports = 1;
    
    const unsigned int num_input = 96*96;
    const unsigned int num_output = 30;
    const unsigned int num_layers = 3;
    const unsigned int num_neurons_hidden = 25;
    
    const float desired_error = (const float) 0.001;
   
    fann_type *calc_out;
    unsigned int i;
    int incorrect,ret = 0;
    int orig,pred; float max =0 ;
    
    
    
    struct fann *ann = fann_create_standard(num_layers, num_input, num_neurons_hidden,
         num_output);

    fann_set_activation_function_hidden(ann, FANN_SIGMOID);
    fann_set_activation_function_output(ann, FANN_LINEAR);

    fann_train_on_file(ann, "facial-train.txt", max_epochs,
        epochs_between_reports, desired_error);

    fann_reset_MSE(ann);
    
    
    
    struct fann_train_data *data = fann_read_train_from_file("facial-test.txt");
    
    printf("Testing network..\n");
    
    for(i = 0; i < fann_length_train_data(data); i++) {
        
        calc_out = fann_test(ann, data->input[i], data->output[i] );
        
        printf ("%i ", i );
       
        max = calc_out[0];
        int maxo = data->output[i][0];
        
        for (int n=0; n<30; n++) {
            printf (" %.2f/%.2f(%.2f) ",calc_out[n], data->output[i][n], data->output[i][n] - calc_out[n]  );
           
            
            
        }
        
        printf ("\n");
        
       
    }
    
    printf("Mean Square Error: %f\n", fann_get_MSE(ann));
    //printf ("Incorrect %i\n", incorrect);
    
    fann_save(ann, "facial.net");

	fann_destroy_train(data);
	fann_destroy(ann);


	
    return 0;
}