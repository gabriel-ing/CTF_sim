import numpy as np
import pandas as pd
import math
import cv2

def get_wavelength(voltage):
    '''
    get the wavelength of an electron beam accelerated to a given voltage, value in kV.
    '''
    #print('Voltage=', voltage)
    planck = 6.62607015e-34
    lightspeed=3e+08
    #electron_rest_mass=  9.1093837015e-31 
    electron_rest_energy = 511000

    kinetic_energy_of_electron=voltage
    electron_charge=1.60217663e-19
    relativistic_corrected_voltage=voltage/0.8366
    #wavelength=1.23e3/((voltage*(1+9.78e-7*voltage))**0.5)
    wavelength=(planck*lightspeed)/((((2*electron_rest_energy*voltage)+(voltage**2))*1.602e-19*1.602e-19)**0.5)
    return wavelength


def create_CTF_data(defocus, voltage, cs, Cc, additional_phase_shift,  amplitude_contrast, angle_of_source, apply_scattering=0,apply_envelope=1, energy_spread=0.7, pixel_size=1, shape=500):
    data = pd.DataFrame()
    voltageV = voltage*1000
    csM = cs*0.001
    CcM = Cc* 0.001
    wavelength=get_wavelength(voltageV)
    additional_phase_shiftRAD=additional_phase_shift*math.pi/180
    defocusM = -defocus*1e-9
    amplitude_contrast_inv = (1-amplitude_contrast*amplitude_contrast)**0.5
    angle_of_sourceRAD=angle_of_source/1000
    #energy_spread = 0.7
#    print(defocusM)


    #additional items
    
    focal_spread = 0 #focal spread in m
    
    SPC_amp1 = 1.494
    SPC_amp2 = 0.937
    SPC_halfwidth1 = 2.322E-19
    SPC_halfwidth2 = 3.79E-20

    max_th = 1/(pixel_size/2) * (1/0.000000001) * wavelength
    #print(max_th)

    
    #data['thPrep']=[x/10000 for x in range(2,1500, 2)]    

    max_th_range = int(max_th*1000000)
    range_step= int(max_th_range/shape)
    #print(max_th_range)
    #print(range_step)
    data['thPrep']=[x/1000000 for x in range(range_step, range_step*2000, range_step)]
    
    
    #print(wavelength)
    data['Theta']=data['thPrep']*wavelength/0.00000000000370155
    #print(wavelength/0.00000000000370155)
    data['Resolution (1/m)']=data['Theta']/wavelength
    data['Resolution (1/nm)'] = data['Resolution (1/m)']*0.000000001
    data['Resolution (nm)'] = 1/data['Resolution (1/nm)']
#    print(csM, wavelength)
    data['Theta*'] = data['Theta']*((csM/wavelength)**0.25)
    
    data['Phase Shift'] = -additional_phase_shiftRAD + (math.pi/2)*(csM*(wavelength**3)*(data['Resolution (1/m)']**4) - 2*defocusM*wavelength*(data['Resolution (1/m)']**2)) #different output
    
    data['sin(W)'] = amplitude_contrast_inv*np.sin(data['Phase Shift'])+amplitude_contrast*np.cos(data['Phase Shift'])
    
    data['Es']= np.exp(-14.238829*((angle_of_sourceRAD**2)*((csM*(wavelength**2)*(data['Resolution (1/m)']**3)-defocusM*data['Resolution (1/m)']))**2))
    
    data['Ed'] = np.exp(-(((math.pi *CcM * wavelength * (data['Resolution (1/m)']**2))**2/11.090355)*((energy_spread/voltageV)**2)))
    
    data['fq'] = ((SPC_amp1*np.exp(-SPC_halfwidth1*(data['Resolution (1/m)']**2))+SPC_amp2*np.exp(-SPC_halfwidth2*(data['Resolution (1/m)']**2))))/2.431
    
    
    data['Et'] = np.exp((-0.5*(np.pi*focal_spread*csM)**2)*(data['Resolution (1/m)']**4))
    
    data['Ec'] = np.exp(-(((math.pi *CcM * wavelength*(data['Resolution (1/m)']**2))/11.090355) * ((energy_spread/voltageV)**2)))
    
    data['Envelope'] = data['Et']*data['Es']*data['Ed']*((1-apply_scattering)+(data['fq']*apply_scattering))*(apply_envelope+(1-apply_envelope))                                     
    
    data['CTF'] = data['Envelope']*data['sin(W)']
    
    data['Power Spectrum']=data['CTF']**2
    data['CTF-Abs'] = abs(data['CTF'])
    #data.head()

    data_2D = make_2D_array(data['Power Spectrum'][:shape])
    return data, data_2D

def make_2D_array(data):
    # Replace this section with your own data generation or data loading code
    num_samples = len(data)  # Number of samples
    your_data = data  # Generate or load your own data here
    
    # Convert the one-dimensional array to a 2D radiating array
    radius = np.linspace(0, 1, num_samples)  # Radial coordinates from 0 to 1
    angle = np.linspace(0, 2 * np.pi, num_samples, endpoint=False)  # Angular coordinates from 0 to 2*pi
    radius_matrix, angle_matrix = np.meshgrid(radius, angle)  # Create a grid of radius and angle
    x = radius_matrix * np.cos(angle_matrix)  # Convert to Cartesian coordinates
    y = radius_matrix * np.sin(angle_matrix)  # Convert to Cartesian coordinates
    indices = (
        np.clip(np.floor((x + 0.5) * num_samples).astype(int), 0, num_samples - 1),
        np.clip(np.floor((y + 0.5) * num_samples).astype(int), 0, num_samples - 1)
    )
    radiating_array = np.zeros((num_samples, num_samples))
    radiating_array[indices] = your_data
    
    # Display the radiating array
    return radiating_array

#df = create_CTF_data(defocus=500, voltage=100, cs=cs, Cc=Cc, additional_phase_shift=additional_phase_shift,amplitude_contrast = amplitude_contrast, angle_of_source=angle_of_source)
#df.head()

def apply_ctf_to_image(data2D, image):
    image = cv2.resize(image, data2D.shape)
    fft = np.fft.fft2(image)
    fshift = np.fft.fftshift(fft) 
    magnitude_spectrum = np.log(np.abs(fshift))
    
    #plt.imshow(magnitude_spectrum)
    fcomplex = fshift*1j
    fshift_filtered = data2D*fcomplex
    f_filtered_shifted = np.fft.fftshift(fshift_filtered) 
    inv_image = np.fft.ifft2(fshift_filtered) 
    
    filtered_image = np.abs(inv_image) 
    filtered_image -= filtered_image.min()
    
    return filtered_image