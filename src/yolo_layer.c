#include "yolo_layer.h"
#include "activations.h"
#include "blas.h"
#include "box.h"
#include "cuda.h"
#include "utils.h"

#include <stdio.h>
#include <assert.h>
#include <string.h>
#include <stdlib.h>




 //                                          4                             3          9         [0, 1, 2]          80
layer make_yolo_layer( int batch, int w, int h, int n, int total, int *mask, int classes )
{

        int i;
        layer l = {0};
        l.type = YOLO;

        l.n = n; // 3
        l.total = total; // 9
        l.batch = batch; // 4
        l.h = h;
        l.w = w;
        l.c = n*(classes + 4 + 1); // 3 * (80 + 4 + 1)
        l.out_w = l.w;
        l.out_h = l.h;
        l.out_c = l.c;
        l.classes = classes; // 80
        l.cost = calloc(1, sizeof(float));
        l.biases = calloc(total*2, sizeof(float));

        if(mask) l.mask = mask; // [0, 1, 2]
        else{
                l.mask = calloc(n, sizeof(int));
                for(i = 0; i < n; ++i){
                l.mask[i] = i;
                }
        }

        l.bias_updates = calloc(n*2, sizeof(float));
        l.outputs = h*w*n*(classes + 4 + 1); // h * w * 3 * (80 + 4 + 1)
        l.inputs = l.outputs;
        l.truths = 90*(4 + 1);
        l.delta = calloc(batch*l.outputs, sizeof(float));
        l.output = calloc(batch*l.outputs, sizeof(float));

        for(i = 0; i < total*2; ++i){
                l.biases[i] = .5;
        }

        l.forward = forward_yolo_layer;
        l.backward = backward_yolo_layer;

#ifdef GPU
        l.forward_gpu = forward_yolo_layer_gpu;
        l.backward_gpu = backward_yolo_layer_gpu;
        l.output_gpu = cuda_make_array(l.output, batch*l.outputs);
        l.delta_gpu = cuda_make_array(l.delta, batch*l.outputs);
#endif

        fprintf(stderr, "yolo\n");
        srand(0);

        return l;

} // layer make_yolo_layer( int batch, int w, int h, int n, int total, int *mask, int classes )



void resize_yolo_layer(layer *l, int w, int h)
{
        l->w = w;
        l->h = h;

        l->outputs = h*w*l->n*(l->classes + 4 + 1);
        l->inputs = l->outputs;

        l->output = realloc(l->output, l->batch*l->outputs*sizeof(float));
        l->delta = realloc(l->delta, l->batch*l->outputs*sizeof(float));

#ifdef GPU
        cuda_free(l->delta_gpu);
        cuda_free(l->output_gpu);

        l->delta_gpu =     cuda_make_array(l->delta, l->batch*l->outputs);
        l->output_gpu =    cuda_make_array(l->output, l->batch*l->outputs);
#endif
}


//                                I.output               I.biases        0       box_index        i       j       I.w      I.h    net->w   net->h    I.w*I.h
box get_yolo_box( float *x,           float *biases,   int n,     int index,      int i, int j, int lw, int lh,   int w,       int h,  int stride )
{
        box b;
        b.x = (i + x[index + 0*stride]) / lw;
        b.y = (j + x[index + 1*stride]) / lh;
        b.w = exp(x[index + 2*stride]) * biases[2*n] / w;
        b.h = exp(x[index + 3*stride]) * biases[2*n+1] / h;
        return b;
}


//                                                I.output      I.biases       0    box_index                       I.w     I.h     net.w   net.h     I.delta    2-truth.w*truth.h          I.w*I.h
float delta_yolo_box( box truth, float *x, float *biases, int n, int index,   int i, int j,   int lw, int lh,  int w,   int h, float *delta,     float scale,           int stride )
{
        box pred = get_yolo_box(x, biases, n, index, i, j, lw, lh, w, h, stride);
        float iou = box_iou(pred, truth);

        float tx = (truth.x*lw - i);
        float ty = (truth.y*lh - j);
        float tw = log(truth.w*w / biases[2*n]);
        float th = log(truth.h*h / biases[2*n + 1]);

        delta[index + 0*stride] = scale * (tx - x[index + 0*stride]);
        delta[index + 1*stride] = scale * (ty - x[index + 1*stride]);
        delta[index + 2*stride] = scale * (tw - x[index + 2*stride]);
        delta[index + 3*stride] = scale * (th - x[index + 3*stride]);
        return iou;
}


void delta_yolo_class(float *output, float *delta, int index, int class, int classes, int stride, float *avg_cat)
{
        int n;
        if (delta[index]){
                delta[index + stride*class] = 1 - output[index + stride*class];
                if(avg_cat) *avg_cat += output[index + stride*class];
                return;
        }
        for(n = 0; n < classes; ++n){
                delta[index + stride*n] = ((n == class)?1 : 0) - output[index + stride*n];
                if(n == class && avg_cat) *avg_cat += output[index + stride*n];
        }
}


static int entry_index(layer l, int batch, int location, int entry)
{
        /* batch = 0, location = n*l.w*l.h + i, entry = 4, could be */
        int n =   location / (l.w*l.h);
        int loc = location % (l.w*l.h);
        return batch*l.outputs + n*l.w*l.h*(4+l.classes+1) + entry*l.w*l.h + loc;
}




/* the essential yolo layer forward funtion */
void forward_yolo_layer(const layer l, network net)
{

        /* need to realize that, net->input is the output of the last layer */

        int i,j,b,t,n;
        memcpy(l.output, net.input, l.outputs*l.batch*sizeof(float)); // I.outputs =  I.h * I.w * 3 * (80 + 4 + 1) = I.inputs, so I.output = net.input

#ifndef GPU
        for (b = 0; b < l.batch; ++b){ // I.batch = 4, could be
                for(n = 0; n < l.n; ++n){ // I.n = 3, could be

                        int index = entry_index(l, b, n*l.w*l.h, 0);
                        /* what this index means could be, 

                        the input have 4 uints (the batch size is 4), for each unit, let's see for the first unit, 
                        for the first unit, we have 3 * (80 + 4 + 1) = I.n * (I.classes + 4 + 1) channels, 
                        for each channel, we have I.w * I.h points, 
                        
                        then we split the 3 * (80 + 4 + 1) channels into 3 groups, and the order shoud be kept, for example [0, 1, 2, 3, 4, 5] splits into [0, 1], [2, 3], [4, 5], 
                        the index is the index for the first point in each group */

                        activate_array(l.output + index, 2*l.w*l.h, LOGISTIC);
                        /* what does the function activate_array do could be, 
                        
                        as contextual description, we split the 3 * (80 + 4 + 1) channels in each unit into 3 group , and each group has (80 + 4 + 1) channels, 
                        then the function activate_array activates the points of 2 channels in the front of each group using the LOGISTIC activation function, 
                        then we get the corresponding oput points. */

                        index = entry_index(l, b, n*l.w*l.h, 4);
                        activate_array(l.output + index, (1+l.classes)*l.w*l.h, LOGISTIC);
                        /* as the contextual description, easy to understand the two statements, 
                        which activates the points of end 81 channels in each group using the LOGISTIC activation funciton. */
                }
        }
#endif

        memset(l.delta, 0, l.outputs * l.batch * sizeof(float));
        if(!net.train) return;

        float avg_iou = 0;
        float recall = 0;
        float recall75 = 0;
        float avg_cat = 0;
        float avg_obj = 0;
        float avg_anyobj = 0;
        int count = 0;
        int class_count = 0;
        *(l.cost) = 0;


        for (b = 0; b < l.batch; ++b) { // I.batch = 4

                for (j = 0; j < l.h; ++j) {

                        for (i = 0; i < l.w; ++i) {

                                for (n = 0; n < l.n; ++n) { // I.n = 3

                                        int box_index = entry_index(l, b, n*l.w*l.h + j*l.w + i, 0);
                                        /* from contextual description we know that, the output of the layer have 4 units (the batch size is 4), 
                                        for each unit we have 3 * (80 + 4 + 1) channels, for each channel we have I.h * I.w points, 
                                        
                                        for each unit we split the 3*(80 + 4 + 1) channels into 3 group, and keep the channels in the original order, 
                                        then the box_index is the index for the (j*l.w + i)th point in the first channel of each group. 
                                        
                                        note added in the 2018-10-04-11-17. */
                        
                                        /* *I.biases = [10, 13,  16, 30,  33, 23,  30, 61,  62, 45,  59, 119,  116, 90,  156, 198,  373, 326] with 18 elements, could be
                                        I.mask = [0, 1, 2] 
                                        i is the x axis coordinate, j is the y axis coordinate */                               
                                        box pred = get_yolo_box( l.output, l.biases, l.mask[n], box_index, i, j, l.w, l.h, net.w, net.h, l.w*l.h );
                                        /* from the function get_yolo_box, we could know, 
                                        the output has 4 unit (the batch size is 4), 
                                        for each unit we have 3 group, for each group we have (80 + 1 +4) channels, for each channel we have I.w * I.h elements, 
                                        
                                        so for the 85 channels in each group, the first 2 channels is the biases for the center point of the boxes, 
                                        and the the third and fourth channels is related to the width and height of of the boxes. 
                                        for the details could check the get_yolo_box function definition. 
                                        
                                        note added in 2018-10-04-12-23. */


                                        float best_iou = 0;
                                        int best_t = 0;
                                        for(t = 0; t < l.max_boxes; ++t){ // I.max_boxes = 90, could be. 
                                                /* l.truths = 90 * (4 + 1) 
                                                net->truth is a pointer to an array with 90 * (4 + 1) * 4 float elements, 
                                                which represents each batch has 4 images, each image has at most 90 boxes, each box represented by 5 float elements (x, y, w, h, obj) */
                                                box truth = float_to_box( net.truth + t*(4 + 1) + b*l.truths, 1 );
                                                if(!truth.x) break;
                                                float iou = box_iou(pred, truth);
                                                if (iou > best_iou) {
                                                        best_iou = iou; // the bigest overlap between the predicated box and true box
                                                        best_t = t;        // the index for the best match true box
                                                }

                                                /* for each image, we have at most 90 truth boxes, the for loop is to 
                                                find out the best match true box, for which the overlap with the predicated box is the bigest */
                                        }


                                        int obj_index = entry_index(l, b, n*l.w*l.h + j*l.w + i, 4);
                                        /* as the contextual description, we could know,  the output has 4 units, 
                                        for each unit we have 3 groups, for each group we have (80 + 1 + 4) channels, for each channel we have I.w * I.h elements, 
                                        so the obj_index is the index for the (j*I.w + i)th element in the 4th channel of the n-th grpup of b-th unit in the batch.
                                        
                                        not added in 2018-10-04-12-48  */

                                        avg_anyobj += l.output[obj_index]; // avg_anyobj is initially 0 
                                        l.delta[obj_index] = 0 - l.output[obj_index];
                                        /* the array which pointer I.delta points to is with the same size of I.output, 
                                        so I.delta.size = 4 * 3 * (80 + 1 + 4) * I.w * I.h */

                                        if (best_iou > l.ignore_thresh) { // I.ignore_thresh = 0.7
                                                l.delta[obj_index] = 0;
                                        }

                                        if (best_iou > l.truth_thresh) { // I.truth_thresh = 1
                                                l.delta[obj_index] = 1 - l.output[obj_index];

                                                // the class is the class of the best match true box
                                                int class = net.truth[best_t*(4 + 1) + b*l.truths + 4]; // the object in the best match true box. 
                                                if (l.map) class = l.map[class]; // I.map = 0

                                                int class_index = entry_index(l, b, n*l.w*l.h + j*l.w + i, 4 + 1);
                                                /* class_index is the index for the (j*I.w + i)th element of the 5th channel in the n-th group of b-th unit */
                                                delta_yolo_class( l.output, l.delta, class_index, class, l.classes, l.w*l.h, 0 ); // I.classes = 80
                                                /* what the function delta_yolo_class do might be,
                                                first let us assume (j*I.w + i)th or (i, j) point in a I.w*I.h table (the table is made to illustrate), 
                                                corresponding to the center of a box for each n, the box is pred which is derived from the I.output (
                                                see pred = get_yolo_box( l.output, l.biases, l.mask[n], box_index, i, j, l.w, l.h, net.w, net.h, l.w*l.h ) for details )

                                                and in the box is an object, which the type for the object could be one of the 80 types, 
                                                so there have 80 possibilities derived in the I.output, 
                                                
                                                then the function delta_yolo_class is to calculate the the difference between the 80 possibilities and the 1 or 0, 
                                                and then used to calculate the total cost of the network.
                                                
                                                note added in 2018-10-05-14-49. */

                                                box truth = float_to_box( net.truth + best_t*(4 + 1) + b*l.truths, 1 );
                                                delta_yolo_box(truth, l.output, l.biases, l.mask[n], box_index, i, j, l.w, l.h, net.w, net.h, l.delta, (2-truth.w*truth.h), l.w*l.h);
                                                /* what the function delta_yolo_box do might be, 
                                                caculate the difference between the predicated box and the best match true box. */
                                        }

                                } // for (n = 0; n < l.n; ++n) {

                        } // for (i = 0; i < l.w; ++i) {

                } // for (j = 0; j < l.h; ++j) {


                for(t = 0; t < l.max_boxes; ++t){ // I.max_boxes = 90

                /* what this for loop do might be, 
                in each batch we have 4 images, each image has at most 90 true boxes, 
                for each true box, finds out the best match predicated box, 
                and then calculate the difference the best predicated box and true box by function delta_yolo_box and function delta_yolo_class. 
                the difference is updated to the I.delta which is used to calculate the network cost for training. 
                
                note added in 2018-10-05-15-38 */

                        box truth = float_to_box(net.truth + t*(4 + 1) + b*l.truths, 1);
                        if(!truth.x) break;

                        float best_iou = 0;
                        int best_n = 0;
                        i = (truth.x * l.w);
                        j = (truth.y * l.h);
                        box truth_shift = truth;
                        truth_shift.x = truth_shift.y = 0;

                        for(n = 0; n < l.total; ++n){ // I.total = 9
                                box pred = {0};
                                pred.w = l.biases[2*n]/net.w;
                                pred.h = l.biases[2*n+1]/net.h;
                                float iou = box_iou(pred, truth_shift);
                                if (iou > best_iou){
                                        best_iou = iou;
                                        best_n = n;
                                }
                        }

                        int mask_n = int_index(l.mask, best_n, l.n); // I.n = 3, I.mask = [0, 1, 2] 
                        if(mask_n >= 0){

                                int box_index = entry_index(l, b, mask_n*l.w*l.h + j*l.w + i, 0);
                                float iou = delta_yolo_box(truth, l.output, l.biases, best_n, box_index, i, j, l.w, l.h, net.w, net.h, l.delta, (2-truth.w*truth.h), l.w*l.h);

                                int obj_index = entry_index(l, b, mask_n*l.w*l.h + j*l.w + i, 4);
                                avg_obj += l.output[obj_index];
                                l.delta[obj_index] = 1 - l.output[obj_index];

                                int class = net.truth[t*(4 + 1) + b*l.truths + 4];
                                if (l.map) class = l.map[class];
                                int class_index = entry_index(l, b, mask_n*l.w*l.h + j*l.w + i, 4 + 1);
                                delta_yolo_class(l.output, l.delta, class_index, class, l.classes, l.w*l.h, &avg_cat);

                                ++count;
                                ++class_count;
                                if(iou > .5) recall += 1;
                                if(iou > .75) recall75 += 1;
                                avg_iou += iou;

                        } // if(mask_n >= 0){

                } // for(t = 0; t < l.max_boxes; ++t){

        } // for (b = 0; b < l.batch; ++b) {


    *(l.cost) = pow( mag_array(l.delta, l.outputs * l.batch), 2 );

    printf("Region %d Avg IOU: %f, Class: %f, Obj: %f, No Obj: %f, .5R: %f, .75R: %f,  count: %d\n", net.index, avg_iou/count, avg_cat/class_count, avg_obj/count, avg_anyobj/(l.w*l.h*l.n*l.batch), recall/count, recall75/count, count);

} // void forward_yolo_layer(const layer l, network net)





void backward_yolo_layer(const layer l, network net)
{
        axpy_cpu(l.batch*l.inputs, 1, l.delta, 1, net.delta, 1);
}



void correct_yolo_boxes(detection *dets, int n, int w, int h, int netw, int neth, int relative)
{
        int i;
        int new_w=0;
        int new_h=0;

        if (((float)netw/w) < ((float)neth/h)) {
                new_w = netw;
                new_h = (h * netw)/w;
        } else {
                new_h = neth;
                new_w = (w * neth)/h;
        }

        for (i = 0; i < n; ++i){
                box b = dets[i].bbox;
                b.x =  (b.x - (netw - new_w)/2./netw) / ((float)new_w/netw); 
                b.y =  (b.y - (neth - new_h)/2./neth) / ((float)new_h/neth); 
                b.w *= (float)netw/new_w;
                b.h *= (float)neth/new_h;
                if(!relative){
                        b.x *= w;
                        b.w *= w;
                        b.y *= h;
                        b.h *= h;
                }
                dets[i].bbox = b;
        }

} // void correct_yolo_boxes(detection *dets, int n, int w, int h, int netw, int neth, int relative)



int yolo_num_detections(layer l, float thresh) // thresh = 0.5
{
        int i, n;
        int count = 0;
        for (i = 0; i < l.w*l.h; ++i){
                for(n = 0; n < l.n; ++n){ // I.n = 3, from the file 'cfg/yolov3.cfg' and function parse_yolo we could know that. 
                        int obj_index  = entry_index(l, 0, n*l.w*l.h + i, 4);
                        /* to understand the obj_index, let us assume, 

                        in the I.output we have 3 group (as for the test time, the batch size is 1, so the unit is 1), 
                        each group corresponding to each kind of boxes, (here we have I.n = 3 kinds of boxes, each kind of boxes has different size), 
                        and each group has (80 + 4 +1) channels, each channel has I.w*I.h float elements, 

                        we also assume that there exists a I.w*I.h table T, for example I.w = 3, I.h = 3, then T = [[0,0], [0,1], [0,2], [1,0], [1,1], [1,2], [2,0], [2,1], [2,2]], 
                        
                        then the channel[0] is the biases for x coordinate of the centers of all the boxes located in the points of T, 
                        the channel[1] is the biases for y coordinate of the centers of all the boxes located in the points of T, 
                        the channel[2] is the widthes parameter of all the boxes located in the points of T, 
                        the channel[3] is the heights parameter of all the boxes located in the points of T, 

                        the channel[4] is the possibilities of all the boxes located in the points of T which actually contains an object,
                        the channel[5] is the possibilities of all the boxes located in the points of T which contains object 1, 
                        the channel[6] is the possibilities of all the boxes located in the points of T which contains object 2, 
                        ...
                        the channel[84] is the possibilities of all the boxes located in the points of T which contains object 80,
                        
                        finally, obj_index is the index for the ith float element in 4th channel in the n-th group of I.output. 
                        
                        note added in 2018-10-05-16-09. */
                        if(l.output[obj_index] > thresh){ // thresh = 0.5
                                ++count;
                        }
                }
        }

        return count;

} // int yolo_num_detections(layer l, float thresh)



void avg_flipped_yolo(layer l)
{

        int i,j,n,z;
        float *flip = l.output + l.outputs;
        for (j = 0; j < l.h; ++j) {
                for (i = 0; i < l.w/2; ++i) {
                        for (n = 0; n < l.n; ++n) {
                                for(z = 0; z < l.classes + 4 + 1; ++z){
                                        int i1 = z*l.w*l.h*l.n + n*l.w*l.h + j*l.w + i;
                                        int i2 = z*l.w*l.h*l.n + n*l.w*l.h + j*l.w + (l.w - i - 1);
                                        float swap = flip[i1];
                                        flip[i1] = flip[i2];
                                        flip[i2] = swap;
                                        if(z == 0){
                                                flip[i1] = -flip[i1];
                                                flip[i2] = -flip[i2];
                                        }
                                }
                        }
                }
        }

        for(i = 0; i < l.outputs; ++i){
                l.output[i] = (l.output[i] + flip[i])/2.;
        }

} // void avg_flipped_yolo(layer l)



//                                            I   im.w    im.h    net->w  net->h             0.5              0              1                         dets
int get_yolo_detections(layer l, int w,   int h,  int netw, int neth, float thresh, int *map, int relative, detection *dets)
{
        int i,j,n;
        float *predictions = l.output;
        if (l.batch == 2) avg_flipped_yolo(l);
        int count = 0;

        for (i = 0; i < l.w*l.h; ++i){

                int row = i / l.w;
                int col = i % l.w;

                for(n = 0; n < l.n; ++n){

                        int obj_index  = entry_index(l, 0, n*l.w*l.h + i, 4);
                        float objectness = predictions[obj_index];
                        if(objectness <= thresh) continue; // thresh = 0.5
                        
                        int box_index  = entry_index(l, 0, n*l.w*l.h + i, 0);
                        dets[count].bbox = get_yolo_box(predictions, l.biases, l.mask[n], box_index, col, row, l.w, l.h, netw, neth, l.w*l.h);
                        dets[count].objectness = objectness;
                        dets[count].classes = l.classes;

                        for(j = 0; j < l.classes; ++j){ // I.classes = 80
                                int class_index = entry_index(l, 0, n*l.w*l.h + i, 4 + 1 + j);
                                float prob = objectness*predictions[class_index];
                                dets[count].prob[j] = (prob > thresh) ? prob : 0;
                        }
                        ++count;
                }
        } // for (i = 0; i < l.w*l.h; ++i){

        correct_yolo_boxes(dets, count, w, h, netw, neth, relative);
        return count;

} // int get_yolo_detections(layer l, int w,   int h,  int netw, int neth, float thresh, int *map, int relative, detection *dets)



#ifdef GPU

void forward_yolo_layer_gpu(const layer l, network net)
{
    copy_gpu(l.batch*l.inputs, net.input_gpu, 1, l.output_gpu, 1);
    int b, n;
    for (b = 0; b < l.batch; ++b){
        for(n = 0; n < l.n; ++n){
            int index = entry_index(l, b, n*l.w*l.h, 0);
            activate_array_gpu(l.output_gpu + index, 2*l.w*l.h, LOGISTIC);
            index = entry_index(l, b, n*l.w*l.h, 4);
            activate_array_gpu(l.output_gpu + index, (1+l.classes)*l.w*l.h, LOGISTIC);
        }
    }
    if(!net.train || l.onlyforward){
        cuda_pull_array(l.output_gpu, l.output, l.batch*l.outputs);
        return;
    }

    cuda_pull_array(l.output_gpu, net.input, l.batch*l.inputs);
    forward_yolo_layer(l, net);
    cuda_push_array(l.delta_gpu, l.delta, l.batch*l.outputs);
}

void backward_yolo_layer_gpu(const layer l, network net)
{
    axpy_gpu(l.batch*l.inputs, 1, l.delta_gpu, 1, net.delta_gpu, 1);
}

#endif

