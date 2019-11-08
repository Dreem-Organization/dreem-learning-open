three_groups_record_description = [
    {
        'path': 'eeg/eeg1', 'fs': 125, 'duration': round(3.5 * 60 * 60)
    },
    {
        'path': 'eeg/eeg2', 'fs': 125, 'duration': round(3.7 * 60 * 60)
    },
    {
        'path': 'eeg/eeg3', 'fs': 125, 'duration': round(3.5 * 60 * 60)
    },
    {
        'path': 'eog/eog1', 'fs': 250, 'duration': round(3.6 * 60 * 60)
    },
    {
        'path': 'eog/eog2', 'fs': 250, 'duration': round(3.8 * 60 * 60)
    },
    {
        'path': 'emg/emg1', 'fs': 500, 'duration': round(3.8 * 60 * 60)
    },
    {
        'path': 'emg/emg2', 'fs': 500, 'duration': round(3.8 * 60 * 60)
    }
]

memmaps_description_nested = {}
memmaps_description_nested['signals'] = [
    {
        'name': 'eeg-eog',
        'signals': [
            {'signals': [{'signals': ['signals/eog/eog1', 'signals/eog/eog2'],
                          'processings': []},
                         {'signals': ['signals/eog/eog1', 'signals/eog/eog2'],
                          'processings': []}

                         ],
             'processings': [{'type': 'resample',
                              'args': {'target_frequency': 100}}]

             },
            {'signals': [{'signals': ['signals/eeg/eeg1', 'signals/eeg/eeg2'],
                          'processings': []},
                         {'signals': ['signals/eeg/eeg1', 'signals/eeg/eeg2'],
                          'processings': []}

                         ], 'processings': [{'type': 'resample',
                                             'args': {'target_frequency': 100}}]

             }

        ],
        'signals_name': ['signal_' + str(i) for i in enumerate(range(8))],
        'processings': [{'type': 'padding',
                         'args': {'padding_duration': 300, 'value': 0}
                         }]
    }

]

memmaps_description_nested['features'] = []

memmaps_description_bis = {'features': [], 'signals': []}
memmaps_description_bis['signals'] = [
    {
        'name': 'eeg-eog',
        'signals': [
            {'signals': [{'signals': ['signals/eog/eog1', 'signals/eog/eog2'],
                          'processings': []},
                         {'signals': ['signals/eog/eog1', 'signals/eog/eog2'],
                          'processings': []}

                         ],
             'processings': [{'type': 'resample',
                              'args': {'target_frequency': 100}}]

             }

        ],
        'processings': [{'type': 'padding',
                         'args': {'padding_duration': 300, 'value': 0}
                         }],
        'signals_name': ['signal_' + str(i) for i in enumerate(range(4))],
    }

]

expected_properties = {
    "eeg-eog": {
        "fs": 100,
        "padding": 300,
        "shape": [
            1284000,
            8
        ]
    }
}

groups_description = {
    "eeg-eog": {
        "fs": 100,
        "padding": 300,
        "shape": [
            3000,
            8
        ],
        "window_length": 3000
    }
}

wrong_frequency_memmaps_description = {'signals': [], 'features': []}
wrong_frequency_memmaps_description['signals'] = [

    {'name': 'eog-emg',
     'signals': ['signals/eeg/eeg1', 'signals/emg/emg1'],
     'processings': []
     }
]

wrong_padding_memmaps_description = {'signals': [

    {'name': 'eog-emg',
     'signals': [
         {
             'signals': ['signals/eeg/eeg1'],
             'processings': [{'type': 'padding',
                              'args': {'padding_duration': 60, 'value': 0}
                              }]

         },
         {
             'signals': ['signals/eeg/eeg1'],
             'processings': [{'type': 'padding',
                              'args': {'padding_duration': 90, 'value': 0}
                              }]

         }],
     'signals_name': ['signal_' + str(i) for i in enumerate(range(2))],
     'processings': []
     }
], 'features': []}

invalid_path_description = {"signals": [
    {
        'name': 'eeg-eog',
        'signals': [
            {'signals': [{'signals': ['signals/eeg/emg1', 'signals/eeg/eeg2'],
                          'processings': []},
                         {'signals': ['signals/eeg/eeg1', 'signals/eeg/eog2'],
                          'processings': []}

                         ], 'processings': [{'type': 'resample',
                                             'args': {'target_frequency': 100}}]

             }

        ],
        'processings': [{'type': 'padding',
                         'args': {'padding_duration': 300, 'value': 0}
                         }],
        'signals_name': ['signal_' + str(i) for i in enumerate(range(2))],
    }

], 'features': []}

invalid_processing_name = {"signals": [
    {
        'name': 'eeg',
        'signals': [
            {'signals': [{'signals': ['signals/eeg/eeg1', 'signals/eeg/eeg2'],
                          'processings': [{'type': 'sobolev_filter',
                                           'args': {'order': 2,
                                                    'frequency_band': [0.1, 5],
                                                    'filter_type': 'butter',
                                                    'forward_backward': True}}]}

                         ], 'processings': [{'type': 'resample',
                                             'args': {'target_frequency': 100}}]

             }

        ],
        'signals_name': ['signal_' + str(i) for i in enumerate(range(2))],
        'processings': [{'type': 'padding',
                         'args': {'padding_duration': 300, 'value': 0}
                         }]
    }

], 'features': []}

invalid_processing_args = {"signals": [
    {
        'name': 'eeg',
        'signals': [
            {'signals': [{'signals': ['signals/eeg/eeg1', 'signals/eeg/eeg2'],
                          'processings': [{'type': 'resample',
                                           'args': {'x': 2,
                                                    'y': [0.1, 5],
                                                    'z': 'butter',
                                                    'forward_backward': True}}]}

                         ], 'processings': [{'type': 'resample',
                                             'args': {'target_frequency': 100}}]

             }

        ],
        'signals_name': ['signal_' + str(i) for i in enumerate(range(2))],
        'processings': [{'type': 'padding',
                         'args': {'padding_duration': 300, 'value': 0}
                         }]
    }

], 'features': []}

augmentation_pipeline_nested = [
    {
        'name': 'eeg-eog',
        'processing': [
            {
                'type': 'kill_channel',
                'args': {'p': 1}
            }
        ]
    }
]

augmentation_pipeline_nested_wrong = [
    {
        'name': 'ekg',
        'processing': [
            {
                'type': 'green_noise',
                'args': {'amplitude': 10, 'p': 1}
            }
        ]
    }
]
