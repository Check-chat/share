import h5py
filename = "chatbot_model.h5"

with h5py.File(filename, "r") as f:
    # List all groups
    print("Keys: %s" % f.keys())
    # for i in range(len(f.keys())):
    #     a_group_key = list(f.keys())[i]

    #     # Get the data
    #     data = list(f[a_group_key])
    #     print(data)
    futures_data = h5['model_weights']  # VSTOXX futures data
    options_data = h5['optimizer_weights']  # VSTOXX call option data
    print(futures_data)
    print(options_data)
    h5.close()
