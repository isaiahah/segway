========================
Segway Toolkit Reference
========================

Description
===========

The Segway Toolkit provides a Python framework to design dynamic Bayesian models for custom 
tasks. Segway can perform training and inference using these user-defined models.

The Segway Toolkit is installed with Segway automatically. 

Workflow
========

The Segway Toolkit describes the model using two files:

  1. A structure file describing the graphical network's nodes and 
  edges. This should be written using the GMTK structure language. The GMTK 
  structure language is described in the 
  `GMTK Documentation <https://github.com/melodi-lab/gmtk/blob/master/documentation.pdf>`_.

  2. A Python file describing the initial parameter settings and Segway 
  commands for training and inference. The classes provided by the Segway Toolkit
  for writing this file are described below. It should contain 3 sections:

    1. Code defining an :py:class:`InputMaster` object and setting its 
    appropriate attributes to describe the model.

    2. A line saving the :py:class:`InputMaster` object to an input file, 
    which describes the parameters in GMTK structure format. 

    3. Code calling Segway to train the defined model and annotate using the 
    trained model. More information on running Segway is available in 
    :ref:`python-interface`.

The `CNVway code <https://github.com/hoffmangroup/cnvway>`_ provides a worked 
example applying the Segway Toolkit to defining, training, and running a new model.

.. todo: other section? flip sentence order? link?

InputMaster Class
=================

Central class storing all parameter information.

.. py:class:: InputMaster

    .. py:attribute:: preamble
        :type: str

    Stores a string which is included as the preamble to the parameter file.

    This is the recommended location for C preprocessor commands such as
    ``#define`` and ``#include`` statements.

    .. py:attribute:: dt
        :type: InlineSection

    Stores the DecisionTrees used in the model.

    Behaves as a dictionary where keys are decision tree names and each value 
    should be set to a :py:class:`DecisionTree` object. 

    .. py:attribute:: name_collection
        :type: InlineSection
    
    Stores the names of distributions and their state names. 
    
    Behaves as a dictionary where keys are distribution names, which 
    can be referenced in the structure file, and each value should be set to 
    a list of that distribution's state names or a :py:class:`NameCollection` 
    object initialized with state names. If a Python list is given, it is 
    converted to a :py:class:`NameCollection` object using 
    :py:meth:`NameCollection.__init__`.

    .. py:attribute:: dirichlet
        :type: InlineSection
    
    Stores Dirichlet distributions used in the model. 
    
    Behaves as a dictionary where keys are decision tree names and each value 
    should be set to a :py:class:`DirichletTable` object. 

    .. py:attribute:: deterministic_cpt
        :type: InlineSection

    Stores deterministic conditional probability tables (CPTs).
    
    Behaves as a dictionary where keys are distribution names, which can 
    be referenced in the structure file, and each value is a 
    :py:class:`DeterministicCPT` object. 

    .. py:attribute:: virtual_evidence
        :type: InlineSection

    Stores virtual evidence used in the model.
    
    Behaves as a dictionary where keys are virtual evidence names, and each
    value is a :py:class:`VirtualEvidence` object.

    .. py:attribute:: dense_cpt
        :type: InlineSection

    Stores dense conditional probability tables (CPTs) used in the model. 
    
    Behaves as a dictionary where keys are distribution names, which can 
    be referenced in the structure file, and each value is a 
    :py:class:`DenseCPT` object.

    .. py:attribute:: mean
        :type: InlineSection

    Stores the mean parameters for named distributions. 
    
    Behaves as a dictionary where keys are distribution names and each
    value should be set to the mean value or a :py:class:`Mean` object 
    initialized with the mean value. If a Python float or list of floats is 
    given, it is converted to a :py:class:`Mean` object using 
    :py:meth:`Mean.__init__`.

    .. py:attribute:: covar
        :type: InlineSection

    Stores the covariance parameters for named distributions.
    
    Behaves as a dictionary where keys are distribution names and each
    value should be set to the covariance value or a :py:class:`Covar` object 
    initialized with the covariance value. If a Python float or list of floats is 
    given, it is converted to a :py:class:`Covar` object using 
    :py:meth:`Covar.__init__`.

    .. py:attribute:: dpmf
        :type: InlineSection

    Stores dense probability mass function (DPMF) objects, which can later be used
    to define Gaussian Mixture models. 
    
    Behaves as a dictionary where keys are distribution names and each value 
    should be set to a :py:class:`DPMF` object. 

    .. py:attribute:: mc
        :type: InlineMCSection
        :value: InlineMCSection(mean = self.mean, covar = self.covar)

    Stores Gaussians acting as mixture components (MC) for a Gaussian mixture
    model.
    
    Behaves as a dictionary where keys are distribution names and each value 
    should be set to an :py:class:`MC` object.

    .. py:attribute:: mx
        :type: InlineMXSection
    
    Store Gaussian mixture (mx) distributions constructed from above-defined mixture 
    components and dense probability mass functions.
    
    Behaves as a dictionary where keys are distribution names, usually 
    corresponding to hidden state names of an emission variable and each value
    is an :py:class:`MX` object.

    .. py:method:: __init__(preamble="")

        Create an :py:class:`InputMaster` object where the ``preamble``
        attribute is set to the provided value and all other attributes
        are empty.

        :param preamble: Preamble to include before all parameters.
        :type preamble: str

    .. py:method:: save(filename)

        Save all parameters to the provided file, for Segway to use in training
        and annotation.

        :param filename: Path to input master file, where results are saved
        :type filename: str
        :returns: None
        :rtype: None
    
Usage example:

.. code-block:: python

    # Create InputMaster object
    input_master = InputMaster()
    # Set parameters
    ...
    # Save to output file
    input_master.save("input.master")


Parameter Classes
=================

Class representing user-defined model parameters.

.. py:class:: DecisionTree

    A container class storing a string representing a decision tree.

    .. py:method:: __init__(tree)

        Create a :py:class:`DecisionTree` object containing the provided
        decision tree.

        :param tree: String representation of a decision tree
        :type tree: str

Usage example:

.. code-block:: python

    # Read a string representation of a tree from a file
    example_tree = open("example_tree.dt").readlines()
    input_master.dt["example_tree"] = DecisionTree(example_tree)


.. py:class:: NameCollection

    A list of names with a specialized string method for writing to the 
    parameter file.

    .. py:method:: __init__(names)

        Create a :py:class:`NameCollection` object containing the provided names.

        :param names: List of names
        :type names: list[str]

Usage example:

.. code-block:: python

    # Create a NameCollection object in the InputMaster 
    # name_collection InlineSection
    input_master.name_collection["labels"] = \
        NameCollection(["label1", "label2"])
    # Alternately, a list will be converted to a NameCollection
    input_master.name_collection["labels"] = ["label1", "label2"]


.. py:class:: DirichletTable

    A Numpy ``ndarray`` describing a Dirichlet distribution, with a specialized
    string method for writing to the parameter file.

    .. py:method:: __init__(*args, keep_shape=False)

        Create a :py:class:`DirichletTable` object storing the provided distribution.

        :param args: The probability distribution as an array of probabilties which is interpreted by the Numpy ``array`` constructor. 
        :type args: array_like
        :param keep_shape: If a single item is passed, this determines if that item's shape is kept or if an additional leading dimension of size 1 is added by the Numpy ``array`` constructor. It has no effect when multiple arguments are passed.
        :type args keep_shape: bool

Usage example:

.. code-block:: python

    # Create a DirichletTable object with the given probability distribution
    input_master.dirichlet["diriclet_prior"] = \
        DirichletTable(
            [[990, 9, 0],
             [990, 9, 0],
             [990, 9, 0]])


.. py:class:: DeterministicCPT

    A deterministic conditional probability table (CPT) described using an 
    existing decision tree with a specialized string method for writing to 
    the parameter file.

    .. py:attribute:: cardinality_parents
        :type: tuple[int]

        A tuple of integers describing the cardinality (number of states) for
        the parent variables. If it is empty, there are no parent variables.

    .. py:attribute:: cardinality
        :type: int

        The cardinality of this variable.

    .. py:attribute:: dt
        :type: str

        The name of the decision tree representing this deterministic CPT.

    .. py:method:: __init__(cardinality_parents, cardinality, dt)
        
        Creates a :py:class:`DeterministicCPT` with the provided attributes.

        :param cardinality_parents: The cardinality of parent variables
        :type cardinality_parents: tuple[int] or int
        :param cardinality: The cardinality of this variable
        :type cardinality: int
        :param dt: Name of an existing decision tree 
        :type dt: str 

Usage example:

.. code-block:: python

    # Create a DeterministicCPT object referencing the example_tree Decision Tree
    input_master.deterministic_cpt["example_deterministic_cpt"] = \
        DeterministicCPT((parent1_card, parent2_card), child_card,
                         "example_tree")


.. py:class:: VirtualEvidence

    A Virtual Evidence object with a specialized string method for writing the
    data to the parameter file.

    .. py:method:: __init__(num_segs, ve_list_filename)

        Create a :py:class:`VirtualEvidence` object which will be printed in
        the format
        ``1 num_segs 2 ve_list_filename nfs:num_segs nis:0 fmt:ascii END``.

        :param num_segs: The number of segments this virtual evidence will cover
        :type num_segs: int
        :param ve_list_filename: filename containing the virtual evience list
        :type ve_list_filename: str

Usage example:

.. code-block:: python

    # Create a VirtualEvidence object over 4 segments referencing "ve_file.txt"
    input_master.virtual_evidence = VirtualEvidence(4, "ve_file.txt")


.. py:class:: DenseCPT

    A Numpy ``ndarray`` representing a dense conditional probability table 
    (CPT) with a specialized string method for writing to the parameter file. 
    Supports up to 3 dimensional tables.
    
    .. py:method:: __init__(*args)

        Create a :py:class:`DenseCPT` object storing the provided distribution.

        :param args: The probability distribution as an array of probabilties which is interpreted by the Numpy ``array`` constructor. 
        :type args: array_like
        :param keep_shape: If a single item is passed, this determines if that item's shape is kept or if an additional leading dimension of size 1 is added by the Numpy ``array`` constructor. It has no effect when multiple arguments are passed.
        :param keep_shape: bool

    .. py:method:: set_dirichlet_table(dirichlet_name)

        Set the name of a :py:class:`DirichletTable` this object will reference as a prior
        in its string representation. If this method is not called, the string
        reference will not reference a :py:class:`DirichletTable` prior.

        :param dirichlet_name: Name of :py:class:`DirichletTable` object to reference as prior
        :type dirichlet_name: str
        :returns: None
        :rtype: None

    .. py:classmethod:: uniform_from_shape(*shape, self_transition=0.0)

        A class method for creating a :py:class:`DenseCPT` object with the provided 
        shape.
        If the table is 2 or 3 dimensional, the diagonal entries of the table 
        are set to the ``self_transition`` parameter (default 0.0) and all other 
        entries are set to be uniform. 

        :param shape: Shape of Dense CPT table
        :type shape: Array_like or multiple arguments
        :param self_transition: Value for diagonal entries in the table. Defaults to 0.0
        :type self_transition: float
        :returns: Uniform :py:class:`DenseCPT` object with the given shape and transition probabilities
        :rtype: :py:class:`DenseCPT`

Usage example:

.. code-block:: python

    # Create a custom DenseCPT in the InputMaster dense_cpt 
    # InlineSection.
    input_master.dense_cpt["start"] = \
        DenseCPT([[0.7, 0.3], [0.8, 0.2]])
    # Create a DenseCPT with specified diagonal value and 
    # uniform other values
    input_master.dense_cpt["transition"] = \
        DenseCPT.uniform_from_shape(2, 2, self_transition = 0.6)


.. py:class:: Mean

    A Numpy ``ndarray`` representing a distribution's mean, with a specialized 
    string method for writing to the parameter file. Supports monovariate 
    and multivariate distributions.

    .. py:method:: __init__(*args)

        Create a :py:class:`Mean` object storing the provided mean value or 
        vector.

        :param args: The mean value which is interpreted by the Numpy ``array`` constructor. 
        :type args: array_like

Usage example:

.. code-block:: python

    # Create a Mean object in the InputMaster mean InlineSection
    input_master.mean["dist1"] = Mean(0.0)
    # Alternately, a numeric value will be converted to a Mean
    input_master.mean["dist2"] = 0.0


.. py:class:: Covar

    A Numpy ``ndarray`` representing a distribution's covariance, with a 
    specialized string method for writing to the parameter file. Supports 
    monovariate and multivariate distributions.

    .. py:method:: __init__(*args)

        Create a :py:class:`Covar` object storing the provided covariance 
        value or vector.

        :param args: The covariance value which is interpreted by the Numpy ``array`` constructor. 
        :type args: array_like

Usage example:

.. code-block:: python

    # Create a Covar object in the InputMaster covar InlineSection
    input_master.covar["dist1"] = Covar(1.0)
    # Alternately, a numeric value will be converted to a Covar
    input_master.covar["dist2"] = 1.0


.. py:class:: DPMF

    A Numpy ``ndarray`` representing a dense probability mass function (DPMF) 
    with a specialized string method for writing to the parameter file. As it 
    is intended for use in Gaussian mixture models, it supports monovariate 
    distributions only. 

    .. py:method:: __init__(*args)

        Create a :py:class:`DPMF` object storing the provided distribution.

        :param args: The probability distribution as an array of probabilties which is interpreted by the Numpy ``array`` constructor. 
        :type args: array_like or multiple arguments

    .. py:method:: set_dirichlet_pseudocount(pseudocount)

        Set the Dirichlet pseudocount which will be used in the DPMF and included in its string representation.

        :param pseudocount: Dirichlet pseudocount to normalize DPMF entries
        :type pseudocount: int

    .. py:classmethod:: uniform_from_shape(shape)

        A class method for creating a uniform DPMF with the specified shape.

        :param shape: The shape of the DPMF, as its integer length.
        :type shape: int
        :returns: DPMF with given shape and uniform probabilities.
        :rtype: DPMF

Usage example:

.. code-block:: python

    # Create a custom DPMF object in the InputMaster mean InlineSection
    input_master.dpmf["biased"] = DPMF([0.7, 0.3])
    # Create a uniform DPMF with a specified shape
    input_master.dpmf["uniform"] = DPMF.uniform_from_shape(2)


.. py:class:: DiagGaussianMC

    A Gaussian distribution with a diagonal covariance matrix, with the type
    ``COMPONENT_TYPE_DIAG_GAUSSIAN``, for use as a mixture component (MC) in a
    Gaussian mixture model.

    .. py:attribute:: mean
        :type: str

        Name of a :py:class:`Mean` object representing the mean of this 
        Gaussian.
    
    .. py:attribute:: covar
        :type: str
        
        Name of a :py:class:`Covar` object representing the covariance 
        vector along the diagonal of the covariance matrix. 

    .. py:method:: __init__(mean, covar)

        Create a :py:class:`DiagGaussianMC` object with the specified mean 
        and covariance.

        :param mean: Name of a Mean object for the distribution mean
        :type mean: str
        :param covar: Name of a Covar object for the diagonal covariance vector of the distribution
        :type covar: str

    
.. py:class:: MissingFeatureDiagGaussianMC

    A Gaussian distribution with a diagonal covariance matrix, with the type
    ``COMPONENT_TYPE_MISSING_FEATURE_SCALED_DIAG_GAUSSIAN``, for use as a
    mixture component (MC) in a Gaussian mixture model.

    .. py:attribute:: mean
        :type: str

        Name of a :py:class:`Mean` object representing the mean of this 
        Gaussian.
    
    .. py:attribute:: covar
        :type: str
        
        Name of a :py:class:`Covar` object representing the covariance 
        vector along the diagonal of the covariance matrix. 

    .. py:method:: __init__(mean, covar)

        Create a :py:class:`MissingFeatureDiagGaussianMC` object with the
        specified mean and covariance.

        :param mean: Name of a Mean object for the distribution mean
        :type mean: str
        :param covar: Name of a Covar object for the diagonal covariance vector of the distribution
        :type covar: str

Usage example:

.. code-block:: python

    # Create a DiagGaussian object in the InputMaster mc 
    # InlineMCSection.
    # Arguments are labels for Mean and Covariance objects.
    input_master.mc["dist1"] = \
        DiagGaussianMC(mean = "dist1", covar = "dist1")
    # Similarly create a MissingFeatureDiagGaussianMC object.
    input_master.mc["dist2"] = \
        MissingFeatureDiagGaussianMC(mean = "dist2", covar = "dist2")


.. py:class:: MX

    A Gaussian mixture (MX) model built from Gaussian mixture components.

    .. py:attribute:: dpmf
        :type: str

        Name of a dense probabiliy mass function :py:class:`DPMF` object 
        representing the contribution of each Gaussian mixture component to 
        the mixture model.
    
    .. py:attribute:: components
        :type: str or list[str]

        Names of Gaussian components associated with the mixture model. 

    .. py:method:: __init__(dpmf, components)

        Create an :py:class:`MX` object with the mixture distribution and 
        components.

        :param dpmf: Name of a DPMF describing mixture weights.
        :type dpmf: str
        :param components: Name or list of names of mixture components
        :type components: str or list[str]

Usage example:

.. code-block:: python

    # Create a MX objects in the InputMaster mx InlineMXSection.
    # Arguments are labels for DPMF and MX objects.
    input_master.mx["emission1"] = MX("uniform", ["dist1", "dist2"])
    input_master.mx["emission2"] = MX("biased", ["dist1", "dist2"])


.. py:class:: GenericString

    A class representing a generic string, with no additional formatting
    performed when converting to a string representation. This is provided for
    representing one-off objects which do not fall into any of the above
    categories. 
    
    Note that if the user wants to use this, they should create a new attribute
    in :py:class:`InputMaster` which is initialized to an :py:class:`InlineSection`
    for ``OBJ_KIND_GENERIC_STRING``.

    .. py:attribute:: contents
        :type: str

        The string which the object will produce as its string representation.

    .. py:method:: __init__(contents)

        Create a :py:class:`GenericString` object with the provided string as
        its contents.

        :param contents: String for this object to provide as its string representation.
        :type contents: str


Internal Classes
================

These classes are internal to the operation of the Segway Toolkit, so the user
should not need to define or interact with these. However, they are documented
here for any developers interested in expanding or customizing the Segway Toolkit.

Section Classes
---------------

Classes to store multiple objects that form one section of the parameter file.
These are used within the ``InputMaster`` object.

.. py:class:: InlineSection

    A type-enforced dictionary with an additional string method for writing 
    to the parameter file. 

    .. py:attribute:: kind
        :type: str or None
        :value: None

        A string denoting the type which can be values in this 
        object. If not given, it is set by the first item. This should not be 
        changed by user. 

.. py:class:: InlineMCSection

    A type-enforced dictionary with an additional string method for writing 
    to the parameter file.

    .. py:attribute:: kind
        :type: str or None
        :value: None

        A string denoting the type which can be values in this 
        object. If not given, it is set by the first item. This should not be 
        changed by user. 

    .. py:attribute:: mean
        :type: InlineSection

        An :py:class:`InlineSection` object storing :py:class:`Mean` objects. 
        The value of ``mean`` parameters in :py:class:`MC` objects should be 
        keys in this object. 

    .. py:attribute:: covar
        :type: InlineSection

        An :py:class:`InlineSection` object storing :py:class:`Covar` objects. 
        The value of ``covar`` parameters in :py:class:`MC` objects should be 
        keys in this object.  

.. py:class:: InlineMXSection

    A type-enforced dictionary with an additional string method for writing 
    to the parameter file.

    .. py:attribute:: kind
        :type: str or None
        :value: None

        A string denoting the type which can be values in this 
        object. If not given, it is set by the first item. This should not be 
        changed by user. 

    .. py:attribute:: dpmf
        :type: InlineSection

        An :py:class:`InlineSection` object storing :py:class:`DPMF` objects. 
        The value of ``dpmf`` parameters in :py:class:`MX` objects should be 
        keys in this object. 


Abstract Parameter Classes
--------------------------

Abstract superclasses of the concrete Parameter classes described above. 

.. py:class:: Array

    An abstract class for array-like data, which inherits from Numpy's ``ndarray`` class.

    The ``__new__`` method (called on creating a new member of the class) is overwritten.
    It verifies the provided arguments have type ``int``, ``float``, or ``ndarray`` before 
    creating a class object containing those arguments. It also accepts the optional
    argument ``keep_shape``, which defaults to false. If 0-dimensional data (a single value)
    is provided and ``keep_shape`` is true, the created object will have 0 dimensions. 
    Otherwise, the created object will have 1 dimension and that value as the only item.

.. py:class:: MultiDimArray

    An abstract class which is a child of :py:class:`Array` and the parent for
    Array-like GMTK parameter classes which have a multiple-line string
    representation such as :py:class:`DenseCPT` and
    :py:class:`DirichletTable`.

    As a child of :py:class:`Array` it behaves like a Numpy ``ndarray`` for
    data storage.

    .. py:method:: __str__()

        Return a string representation of the array, without brackets.

        :returns: String representation of the array, across multiple lines
        :rtype: str

    .. py:method:: get_header_info()

        Return header information regarding the size of the array, formatted
        as as the number of parents (1 less than the number of array
        dimensions) followed by the shape of the array along each dimension,
        with all items separated by spaces.

        This method may be overwritten in subclasses.

        :returns: String representation of the array dimensions for usage as a header
        :rtype: str

.. py:class:: OneLineArray
    
    An abstract class which is a child of :py:class:`Array` and the parent for
    Array-like GMTK parameter classes which have a single-line string
    representation such as :py:class:`Mean`, :py:class:`Covar`, and
    :py:class:`DPMF`.

    As a child of :py:class:`Array` it behaves like a Numpy ``ndarray`` for
    data storage.

    .. py:method:: get_header_info()

        Return both header information regarding the size of the array and the
        array contents, formatted in a single line. The header information is
        a single value describing the number of items in the array. All items
        are separated by spaces.

        This method may be overwritten in subclasses.

        :returns: String representation of the array dimensions for usage as a header
        :rtype: str

.. py:class:: Section

    A type-enforced dictionary with an additional string method for writing 
    to the parameter file. 

    .. py:attribute:: kind
        :type: str or None
        :value: None

        A string denoting the type which can be values in this 
        object. If not given, it is set by the first item. This should not be 
        changed by user.

    .. py:attribute:: line_before
        :type: str

        A string which will be printed immediately before the contents of 
        this section. This is provided to support C preprocessor commands
        (such as ``if`` conditional statements).
        
    .. py:attribute:: line_after
        :type: str

        A string which will be printed immediately after the contents of 
        this section. This is provided to support C preprocessor commands
        (such as ``else`` conditional statments).

    .. py:method:: __init__(kind)

        Create a Section object with the specified kind.

        :param kind: The kind for this section. All dictionary values in the section must have this string as their ``kind`` attribute.
        :type kind: str or None

