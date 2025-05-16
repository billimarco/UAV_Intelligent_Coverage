import numpy as np
import math

class CommunicationChannel:

    def __init__(self, args):
        """
        Inizializza un canale UAV-Ground Unit (GU) con i parametri di propagazione del segnale.
        Gli argomenti vengono passati tramite il parser.
        
        :param args: Gli argomenti ottenuti dal parser
        """
        # Parametri passati dal parser
        self.a = args.a
        self.b = args.b
        self.nNLos = args.nlos_loss
        self.nLos = args.los_loss
        self.rate_of_growth = args.rate_of_growth
        self.transmission_power = args.transmission_power
        self.channel_bandwidth = args.channel_bandwidth
        self.power_spectral_density_of_noise = args.noise_psd
        
    # calculate the Probability of LoS link between one UAV and one GU
    def get_PLoS(self, distance_uav_gu: float, uav_altitude: float):
        elevation_angle = math.degrees(math.asin(uav_altitude / distance_uav_gu))
        return 1 / (1 + self.a * math.exp((-1) * self.b * (elevation_angle - self.a)))

    def get_transition_matrix(self, relative_shift: float, PLoS: float):
        PLoS2NLoS = 2 * ((1 - PLoS) / (1 + math.exp(self.rate_of_growth * relative_shift)) - (1 - PLoS) / 2)  # g1
        PNLoS2LoS = 2 * (PLoS / (1 + math.exp(self.rate_of_growth * relative_shift)) - PLoS / 2)  # g2
        return np.array([
            [1 - PLoS2NLoS, PLoS2NLoS],
            [PNLoS2LoS, 1 - PNLoS2LoS]
        ])


    # calculate the Free Space PathLoss of the link between one UAV and one GU in dB
    # 38.4684 is according to Friis equation with carrier frequency fc = 2GHz
    def get_free_space_PathLoss(self, distance_uav_gu: float):
        return 20 * math.log(distance_uav_gu, 10) + 38.4684


    # calculate PathLoss of the link between one UAV and one GU in dB
    def get_PathLoss(self, distance_uav_gu: float, current_state: int):
        FSPL = self.get_free_space_PathLoss(distance_uav_gu)
        if current_state == 0:
            return FSPL + self.nLos
        return FSPL + self.nNLos

    # def getSINR(path_loss: float, interference_path_loss: []):
    #     return W2dB((dBm2Watt(TRASMISSION_POWER) * getChannelGain(path_loss)) / (
    #                 getInterference(interference_path_loss) + dBm2Watt(
    #             POWER_SPECTRAL_DENSITY_OF_NOISE) * CHANNEL_BANDWIDTH))


    def getSINR(self, path_loss: float, interference_path_loss):
        return self.W2dB((self.dBm2Watt(self.transmission_power) * self.getChannelGain(path_loss)) / (self.dBm2Watt(
                self.power_spectral_density_of_noise) * self.channel_bandwidth))


    def getChannelGain(self, path_loss: float) -> float:
        return 1 / self.dB2Linear(path_loss)


    def getInterference(self,interference_path_loss) -> float:
        interference = 0.0
        for path_loss in interference_path_loss:
            interference += self.dBm2Watt(self.transmission_power) * self.getChannelGain(path_loss)
        return interference

    def get_distance_from_SINR_closed_form(self, target_sinr_db: float, current_state: int):
        """
        Calcola la distanza massima UAV-GU che soddisfa il SINR specificato, usando formula chiusa.
        Valido solo senza interferenze.
        
        :param target_sinr_db: SINR minimo accettabile (in dB)
        :param current_state: 0 = LoS, 1 = NLoS
        :return: distanza (in metri)
        """
        noise_watt = self.dBm2Watt(self.power_spectral_density_of_noise) * self.channel_bandwidth
        noise_db = self.W2dB(noise_watt)
        
        additional_loss = self.nLos if current_state == 0 else self.nNLos
        
        # Pathloss totale massimo ammissibile
        pl_max = self.transmission_power - target_sinr_db - noise_db
        
        # FSPL senza il loss addizionale
        fspl = pl_max - additional_loss

        exponent = (fspl - 38.4684) / 20
        distance = 10 ** exponent

        return distance

    def dB2Linear(self, decibel_value: float):
        return 10 ** (decibel_value / 10)


    def dBm2Watt(self, dBm_value: float):
        return math.pow(10, (dBm_value - 30) / 10)


    def W2dB(self, watt_value: float):
        return math.log(watt_value, 10) * 10
