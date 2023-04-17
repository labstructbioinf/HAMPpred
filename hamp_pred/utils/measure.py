
import os, sys
import math
import numpy as np
from sympy import nsimplify

# get from https://github.com/labstructbioinf/samcc_turbo
sys.path.append('/home/nfs/sdunin/scr/samcc_turbo')
from samcc.bundleClass import bundleClass
from samcc.helper_functions import gen_expected_crick_angles

from biopandas.pdb import PandasPdb

crangles = {'a':19.5,'b':122.35,'c':-134.78,'d': -31.92,'e':70.92 ,'f':173.78,'g':-83.35}


def renumber_hamp_pdb(IN_PDB, OUT_PDB, INDEX=274):
	ppdb = PandasPdb().read_pdb(IN_PDB)
	
	ppdb.df['ATOM'].at[911:, 'chain_id'] = 'B'

	for chain_name, chain_df in ppdb.df['ATOM'].groupby('chain_id'):
	
		res_id = INDEX
	
		for res_name, res_df in chain_df.groupby('residue_number'):
			ppdb.df['ATOM'].at[res_df.index,'residue_number'] = res_id
			res_id += 1
			
	ppdb.to_pdb(path=OUT_PDB, 
				records=['ATOM'], 
				gz=False, 
				append_newline=False)

def fix_md_pdb(PATH_BENCH, PATH_BENCH1, FORCE=False):
	md_files = list()
	for directory in os.listdir(PATH_BENCH):
		print(directory)
		directory  = os.path.join(PATH_BENCH, directory)
		directory1 = directory.replace(PATH_BENCH, PATH_BENCH1)
		dir_files = os.listdir(directory)
		dir_files = [os.path.join(directory, d) for d in dir_files]
		for file in dir_files:
			if not file.endswith('pdb'):
				continue
			file_out = file.replace(PATH_BENCH, PATH_BENCH1)
			
			if os.path.isfile(file_out) and not FORCE: 
				md_files.append(file_out)
				continue
			
			with open(file, 'rt') as f:
				content = f.readlines()
				content = content[1:]
				content = ''.join(content)
			
				if not os.path.isdir(directory1):
					os.mkdir(directory1)
				
				with open(file_out, 'wt') as fo:
					fo.write(content)
				md_files.append(file_out)
	return md_files
	
def measure_one_HAMP(path_hamp, a1_start=None, a1_stop=None, a2_start=None, a2_stop=None, chain1=None, chain2=None):
    
    '''
    calculate HAMP protein descriptors
    params:
        path_hamp (str) path to .pdb file
        a1_start (int) 1st chain hamp motif pdb start index
        a1_stop (int) 1st chain hamp motif pdb stop index
        a2_start (int) 2nd chain hamp motif pdb start index
        a2_stop (int) 2nd chain hamp motif pdb stop index
        chain1 (str) one letter pdb 1st chain name
        chain2 (str) one letter pdb 2nd chain name
    return:
        bundle_df (pd.DataFrame) detailed description of HAMP
        n_crick (np.array) N crick angles per residue, also in bundle_df
        c_crick (np.array) C crick angles per residue, also in bundle_df
    '''
    chain1_range = range(a1_start, a1_stop)
    chain2_range = range(a2_start, a2_stop)
    
    defdata = [
        [chain1_range, chain2_range, chain1_range, chain2_range], 
        [chain1,       chain1,       chain2,       chain2], 
        [False,        False,        False,        False], # False for parallel orientation
        ['x',          'x',          'x',          'x']
    ] #można olać
    
    bundle = bundleClass()
    bundle.from_defdata(path_hamp, *defdata)
    bundle.calc_bundleaxis()
    bundle.calc_crick()
    bundle.calc_radius() 
    bundle.calc_periodicity() 
    bundle.calc_crickdev(P=3.5, REP=7, optimal_ph1=19.5)
    bundle_df = bundle.gendf() 

    crick = bundle_df.crick.values
    n_crick = crick[0::2]
    c_crick = crick[1::2]
    
    #n_crick = (crick[0::4] + crick[2::4]) / 2
    #c_crick = (crick[1::4] + crick[3::4]) / 2

    return bundle_df, n_crick, c_crick

	
def adjustangle(angle):

    if abs(angle)>180 and angle > 0:
        angle = angle - 360.0
    elif abs(angle)>180 and angle < 0:
        angle += 360.0
    return angle
    

def average_angles(angles):
    """Average (mean) of angles

    Return the average of an input sequence of angles. The result is between
    ``0`` and ``2 * math.pi``.
    If the average is not defined (e.g. ``average_angles([0, math.pi]))``,
    a ``ValueError`` is raised.
    """

    x = sum(math.cos(np.radians(a)) for a in angles)
    y = sum(math.sin(np.radians(a)) for a in angles)

    if x == 0 and y == 0:
        raise ValueError(
            "The angle average of the inputs is undefined: %r" % angles)

    # To get outputs from -pi to +pi, delete everything but math.atan2() here.
    return np.degrees(math.fmod(math.atan2(y, x) + 2 * math.pi, 2 * math.pi))



def calcPER(P, tolerance = 0.01):
    f = nsimplify(P, tolerance=tolerance)
    return f.p, f.q

def calcPER_dynamic(P, start=0.005, increase=0.005):
    t = start
    while True:
        rep, turn = calcPER(P, tolerance=t)
        if rep<=29:
            break
        elif t<0.035:
            t+=increase
        else:
            break
    return rep, turn
    
def get_ref_crick(P, start_phi):

    REP_LEN, _ = calcPER(P)

    ref = np.tile(
        np.repeat(gen_expected_crick_angles(P, REP_LEN, start_phi), 2),
        3
    )

    return ref
