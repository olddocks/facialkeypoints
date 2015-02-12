#include "fann.h"
#include <OpenCL/cl.h>

int main()
{
    const unsigned int max_epochs = 100;
    const unsigned int epochs_between_reports = 10;
    
    const unsigned int num_input = 256;
    const unsigned int num_output = 10;
    const unsigned int num_layers = 3;
    const unsigned int num_neurons_hidden = 128;
    const float desired_error = (const float) 0.001;
   
    fann_type *calc_out;
    unsigned int i;
    int incorrect,ret = 0;
    int orig,pred; float max =0 ;
    
    
    
    struct fann *ann = fann_create_standard(num_layers, num_input,
        num_neurons_hidden, num_output);

    fann_set_activation_function_hidden(ann, FANN_SIGMOID);
    fann_set_activation_function_output(ann, FANN_SIGMOID);

    fann_train_on_file(ann, "mnist-train.txt", max_epochs,
        epochs_between_reports, desired_error);

    fann_reset_MSE(ann);
    
    
    
    struct fann_train_data *data = fann_read_train_from_file("mnist-test.txt");
    
    printf("Testing network..\n");
    
    for(i = 0; i < fann_length_train_data(data); i++) {
        
        calc_out = fann_test(ann, data->input[i], data->output[i]);
        
        printf ("Test %i ", i );
        
        max = calc_out[0];
        int maxo = data->output[i][0];
        
        for (int n=0; n<10; n++) {
            printf (" %f ",calc_out[n] );
            if (calc_out[n] > max)
            {
                max = calc_out[n];
                pred = n;
                
            }
            
            
            if (data->output[i][n] > maxo ) {
                
                orig = n;
            }
            
            
            
        }
        
        if ( pred != orig ) {
            incorrect ++;
        }
        
        printf("Orig: %i Predicted: %i -> %f \n",orig, pred, max );
        /*
         printf("%i (%f) -> %f, should be %f, difference=%f", n, data->output[i][n], round(calc_out[n]), data->output[i][n],
         (float) fann_abs(calc_out[n] - data->output[i][n] ) );
         */
    }
    
    printf("Mean Square Error: %f\n", fann_get_MSE(ann));
    printf ("Incorrect %i\n", incorrect);
    
    fann_save(ann, "mnist.net");

	fann_destroy_train(data);
	fann_destroy(ann);


	
    return 0;
}