import Core.baseFunctions
import numpy as np

class LayerEmbed:
    def __init__(self, list_sizes, list_embedding):
        self.list_sizes = list_sizes
        self.list_embedding = list_embedding
        self.embedding_matrix = []
        self.initialize()
        self.activ = "none"
        self.input_size = np.sum(self.list_sizes)
        self.output_size = np.sum(list_embedding) + np.sum(list_sizes[np.where(list_embedding == 0)])
        
    def initialize(self):
      for i,embedding in enumerate(self.list_embedding):
        if embedding > 0:
          input_size = self.list_sizes[i]
          output_size = embedding
          self.embedding_matrix.append((np.random.rand(input_size, output_size) - .5)/np.sqrt(input_size))
        else:
          self.embedding_matrix.append(np.zeros((0,0)))
    
    def fprop(self, input):
      """ We want to store the output for the computation of the gradient. """
      temp_output = np.zeros((input.shape[0], 0))
      column = 0
      for i,embedding in enumerate(self.list_embedding):
        if embedding > 0:
          temp_output = np.hstack((temp_output, np.dot(input[:, column:column+self.list_sizes[i]], self.embedding_matrix[i])))
        else:
          temp_output = np.hstack((temp_output, input[:, column:column+self.list_sizes[i]]))
        column += self.list_sizes[i]
        
      self.output = temp_output

    def get_gradient_sufficient_statistics(self, grad_output_after, input):
        grad_output_before = grad_output_after * base_functions.gradActiv(self, self.output)
        gradient_sufficient_statistics = []
        gradient_sufficient_statistics.append(grad_output_before)
        gradient_sufficient_statistics.append(input)
        return gradient_sufficient_statistics

    def get_gradient_from_sufficient_statistics(self, gradient_sufficient_statistics, input = None):
      grad_output_before = gradient_sufficient_statistics
      gradient = []
      column = 0
      n_data = grad_output_before.shape[0]
      for i,embedding in enumerate(self.list_embedding):
        if embedding > 0:
          grad = np.dot(input[:, column:column+self.list_sizes[i]].T, grad_output_before[:, column:column+self.list_embedding[i]]) / n_data
          gradient.append(grad)
        else:
          gradient.append(np.zeros((0,0)))
        column += self.list_sizes[i]
          
      return gradient
        
    def bprop(self, grad_output_before):
        raise NotImplementedError

    def update(self, gradient, l1_regularizer, l2_regularizer, stepsize, n_seen):
      for i,embedding in enumerate(self.list_embedding):
        if embedding > 0:
          self.embedding_matrix[i] -= gradient[i]*stepsize/n_seen
          if l2_regularizer > 0:
            self.embedding_matrix[i] *= (1 - l2_regularizer*stepsize)
          if l1_regularizer > 0:
            self.embedding_matrix[i]  *= np.fmax(0, 1 - l1_regularizer*stepsize/np.abs(self.embedding_matrix[i] ))

    def compute_regularize(self, l1_regularizer, l2_regularizer):
        loss = 0
        for i,embedding in enumerate(self.list_embedding):
          if embedding > 0:
            if l2_regularizer > 0:
              loss += np.sum(self.embedding_matrix[i]**2) * 0.5 * l2_regularizer
            if l1_regularizer > 0:
              loss += np.sum(np.abs(self.embedding_matrix[i])) * l1_regularizer
        return loss
