def compute_acc(model, data, labels, num_samples=None, batch_size=100):
	"""
	Compute the accuracy of given data and labels

	Arguments:
	- data: Array of input data, of shape (batch_size, d_1, ..., d_k)
	- labels: Array of labels, of shape (batch_size,)
	- num_samples: If not None, subsample the data and only test the model
	  on these sampled datapoints.
	- batch_size: Split data and labels into batches of this size to avoid using
	  too much memory.

	Returns:
	- accuracy: Scalar indicating fraction of inputs that were correctly
	  classified by the model.
	"""
	N = data.shape[0]
	if num_samples is not None and N > num_samples:
		indices = np.random.choice(N, num_samples)
		N = num_samples
		data = data[indices]
		labels = labels[indices]

	num_batches = N // batch_size
	if N % batch_size != 0:
		num_batches += 1
	preds = []
	for i in range(num_batches):
		start = i * batch_size
		end = (i + 1) * batch_size
		output = model.forward(data[start:end], False)
		scores = softmax(output)
		pred = np.argmax(scores, axis=1)
		preds.append(pred)
	preds = np.hstack(preds)
	accuracy = np.mean(preds == labels)
	return accuracy