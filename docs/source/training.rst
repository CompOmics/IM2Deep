Training
========

IM2Deep can train a new CCS prediction model from scratch, or fine-tune an
existing one on your own runs. Both go through the same DeepLC featuriser that
:func:`im2deep.core.predict` uses, so a model that comes out of training is
directly usable for prediction.

Training data
-------------

Training data is a delimited file, a :class:`pandas.DataFrame`, or a
:class:`~psm_utils.psm_list.PSMList`.

Tabular input needs a target CCS column, named either ``CCS`` or ``ccs``, plus
either a ``peptidoform`` column of ProForma strings or the three columns
``seq``, ``modifications`` and ``charge``:

.. code-block:: text

    seq,modifications,charge,ccs
    AAAAAAAAAAAAAAAAASAGGK,,2,445.04
    AAAAAAAAAAAAAAAAAGATCLER,21|U:4,3,634.92

A :class:`~psm_utils.psm_list.PSMList` needs a target on every PSM, as either
``psm.ion_mobility`` or ``psm.metadata["CCS"]``. Ion mobility is converted to
CCS automatically.

Peptides longer than the 60-residue featurisation window are dropped rather than
silently truncated, as are rows with a missing or non-finite CCS. Both are
reported in the log.

Training a new model
--------------------

From the command line:

.. code-block:: bash

    im2deep train train.csv -o my_model.ckpt --epochs 100 --num-workers 16

Or from Python:

.. code-block:: python

    from im2deep import train

    model = train("train.csv", "my_model.ckpt", training_kwargs={"epochs": 100})

A validation set is split off the training data, grouped by stripped sequence so
that the same peptide cannot appear in both halves at different charge states or
modification states. Pass ``--validation-data`` (or ``validation_psm_list=``) to
supply one explicitly instead.

``--num-workers`` matters on large datasets: featurisation happens on the fly in
the dataloader and is CPU-bound, so it is usually the bottleneck rather than the
GPU.

Fine-tuning an existing model
-----------------------------

Fine-tuning adapts a trained model to new data, which is useful when your runs
sit on a systematically different CCS scale than the training data:

.. code-block:: bash

    im2deep finetune my_runs.csv -o finetuned.ckpt --freeze-epochs 5

The pretrained feature branches are held frozen for the first ``--freeze-epochs``
epochs while the head adapts, then unfrozen and training continues at a reduced
learning rate. By default the bundled IM2Deep model is the backbone; pass
``--backbone`` to use another checkpoint.

Configuration
-------------

Every model and training parameter can be set through a JSON configuration file:

.. code-block:: bash

    im2deep train train.csv -o my_model.ckpt --config config.json

.. code-block:: json

    {
      "model_params": {
        "epochs": 100,
        "batch_size": 512,
        "learning_rate": 0.001,
        "patience": 10,
        "wandb": {"enabled": true, "project_name": "IM2Deep"}
      }
    }

The ``model_params`` block is optional; a flat object works too. Values are
merged over :data:`im2deep.constants.DEFAULT_TRAINING_CONFIG`, and explicit
command-line flags or ``training_kwargs`` win over the file.

Weights & Biases logging needs the optional extra:

.. code-block:: bash

    pip install im2deep[wandb]

Featurisation variants
----------------------

Three configuration keys change how peptides are encoded. They alter the number
of global features, so ``Global_features`` must be set to match; training refuses
to start otherwise rather than failing later with a shape error.

.. list-table::
   :header-rows: 1

   * - ``add_ccs_features``
     - ``add_terminal_composition``
     - ``Global_features``
   * - ``false``
     - ``false``
     - 55
   * - ``true`` (default)
     - ``false`` (default)
     - 60
   * - ``false``
     - ``true``
     - 67
   * - ``true``
     - ``true``
     - 72

``legacy_positional_deltas`` controls positional modification indexing. It
defaults to ``true``, reproducing the encoding the bundled models were trained
with. Setting it to ``false`` uses DeepLC's corrected indexing, which changes the
encoding of modified peptides only.

Each trained checkpoint records the configuration it was trained with, so
:func:`im2deep.core.predict` reads a model back with the right architecture and
featurisation automatically. There is no need to repeat these settings at
prediction time.
