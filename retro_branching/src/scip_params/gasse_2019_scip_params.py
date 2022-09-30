gasse_2019_scip_params = {'separating/maxrounds': 0, # separate (cut) only at root node
                          'presolving/maxrestarts': 0, # disable solver restarts
                          'limits/time': 20*60, # solver time limit
                          'timing/clocktype': 2,
                          'limits/gap': 3e-4,
                          'limits/nodes': -1} 