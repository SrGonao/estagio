import pandas
import numpy as np

def moveToCartesian(inData, particle, z = True):
    pt = inData.loc[inData.index[:], particle + "_pT"]
    if z: 
        eta = inData.loc[inData.index[:], particle + "_eta"]  
    phi = inData.loc[inData.index[:], particle + "_phi"]
    
    inData[particle + '_px'] = pt*np.cos(phi)
    inData[particle + '_py'] = pt*np.sin(phi)
    if z: 
        inData[particle + '_pz'] = pt*np.sinh(eta)

        
def moveToPtEtaPhi(inData, particle):
    px = inData.loc[inData.index[:], particle + "_px"]
    py = inData.loc[inData.index[:], particle + "_py"]
    if 'mPT' not in particle: 
        pz = inData.loc[inData.index[:], particle + "_pz"]
        
    inData[particle + '_pT'] = np.sqrt(np.square(px)+np.square(py))
    if 'mPT' not in particle: 
        inData[particle + '_eta'] = np.arcsinh(pz/inData.loc[inData.index[:], particle + '_pT'])
        
    inData[particle + '_phi'] = np.arcsin(py/inData.loc[inData.index[:], particle + '_pT'])
    
    inData.loc[(inData[particle + "_px"] < 0) & (inData[particle + "_py"] > 0), particle + '_phi'] = \
            np.pi - inData.loc[(inData[particle + "_px"] < 0) & (inData[particle + "_py"] > 0), particle + '_phi']
        
    inData.loc[(inData[particle + "_px"] < 0) & (inData[particle + "_py"] < 0), particle + '_phi'] = \
            -1 * (np.pi + inData.loc[(inData[particle + "_px"] < 0) & (inData[particle + "_py"] < 0), particle + '_phi'])
                  
    inData.loc[(inData[particle + "_px"] < 0) & (inData[particle + "_py"] == 0), particle + '_phi'] = \
            np.random.choice([-1*np.pi, np.pi], inData[(inData[particle + "_px"] < 0) & (inData[particle + "_py"] == 0)].shape[0])

    
def deltaphi(a, b):
    return np.pi - np.abs(np.abs(a-b) - np.pi)


def twist(dphi, deta):
    return np.arctan(np.abs(dphi/deta))


def addAbsMom(inData, particle, z=True):
    if z:
        inData[particle + '_|p|'] = np.sqrt(np.square(inData.loc[inData.index[:], particle + '_px']) +
                                            np.square(inData.loc[inData.index[:], particle + '_py']) +
                                            np.square(inData.loc[inData.index[:], particle + '_pz']))
    else:
        inData[particle + '_|p|'] = np.sqrt(np.square(inData.loc[inData.index[:], particle + '_px']) +
                                            np.square(inData.loc[inData.index[:], particle + '_py']))

def addEnergy(inData, particle):
    if particle + '_|p|' not in inData.columns:
        addAbsMom(inData, particle)
        
    inData[particle + '_E'] = np.sqrt(np.square(inData.loc[inData.index[:], particle + '_mass']) +
                                      np.square(inData.loc[inData.index[:], particle + '_|p|']))

def addMT(inData, pT, phi, name):
    inData[name + '_mT'] = np.sqrt(2 * pT * inData['mPT_pT'] * (1 - np.cos(deltaphi(phi, inData['mPT_phi']))))

def addHighLvl(inData):
    finalStates = ['b_0', 'b_1', 't_0', 't_1']
    recoObjects = ['diH', 'h_bb', 'h_tt']
    variables = ['px', 'py', 'pz']
    for fs1 in finalStates:
        for fs2 in finalStates:
            if fs1 == fs2: continue
            for var in variables:
                inData['hl_d' + var + '_' + fs1 + '_' + fs2] = inData.loc[:, fs1 + '_' + var] - inData.loc[:, fs2 + '_' + var]
        inData['hl_dpx_' + fs1 + '_mPT'] = inData.loc[:, fs1 + '_px'] - inData.loc[:, 'mPT_px']
        inData['hl_dpy_' + fs1 + '_mPT'] = inData.loc[:, fs1 + '_py'] - inData.loc[:, 'mPT_py']
        inData[fs1 + '_|p|'] = np.sqrt(np.square(inData.loc[:, fs1 + '_px'])+np.square(inData.loc[:, fs1 + '_py'])+np.square(inData.loc[:, fs1 + '_pz']))
        inData[fs1 + '_E'] = np.sqrt(np.square(inData.loc[:, fs1 + '_mass'])+np.square(inData.loc[:, fs1 + '_|p|']))
    for fs1 in recoObjects:
        for fs2 in recoObjects:
            if fs1 == fs2: continue
            for var in variables:
                inData['hl_d' + var + '_' + fs1 + '_' + fs2] = inData.loc[:, fs1 + '_' + var] - inData.loc[:, fs2 + '_' + var]
        inData['hl_dpx_' + fs1 + '_mPT'] = inData.loc[:, fs1 + '_px'] - inData.loc[:, 'mPT_px']
        inData['hl_dpy_' + fs1 + '_mPT'] = inData.loc[:, fs1 + '_py'] - inData.loc[:, 'mPT_py']
    




def fixData(inData):
    if not inData['gen_target'][0]:
        inData.rename(columns={'weight': 'gen_weight'}, inplace=True)
        rename = [var for var in inData.columns if str.endswith(var, "_e")]
        inData.rename(columns=dict(zip(rename, [var[:var.rfind("_e")] + "_E" for var in rename])), inplace=True)
    rename = [var for var in inData.columns if str.startswith(var, "gen_hh")]
    inData.rename(columns=dict(zip(rename, ["gen_diH" + var[6:] for var in rename])), inplace=True)
    inData.rename(columns={'gen_m_hh' : 'gen_diH_mass'}, inplace=True)