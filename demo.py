import mnist_loader
import network2
training_data,validation_data,test_data=mnist_loader.load_data_wrapper()
# print("training data")
# print(type(training_data))
# print(len(training_data))
# print(training_data[0][0].shape)
# print(training_data[0][1].shape)
#
# print('validation data')
# print(len(validation_data))

# print('test data')
# print(len(test_data))
# net.large_weight_initializer()
net=network2.Network([784,30,10],cost=network2.CrossEntropyCost)
net.SGD(training_data,30,10,0.1,evaluation_data=validation_data,lmbda=5.0,monitor_training_cost=True)