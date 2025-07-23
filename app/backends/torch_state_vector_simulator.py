import logging

import numpy as np
import torch
import torch.nn as nn


logger = logging.getLogger(" [Backends]")


class StateVecSimTorch(nn.Module):
    def __init__(self, n_qubits,
                 n_layers,
                 meas_qubits=None,
                 init_weights_scale=1.,
                 gpu=False,
                 seed=42,
                 ):
        super().__init__()
        torch.manual_seed(seed)
        self.n_qubits = n_qubits
        self.state = None
        self.n_layers = n_layers
        self.meas_qubits = meas_qubits
        if gpu:
            # make ready for gpu
            try:
                self.device = torch.device("cuda")
            except:
                self.device = torch.device("cpu")
        else:
            self.device = torch.device("cpu")

        logger.debug(f'Initializing QML model on device: {self.device}..')

        self.weights = nn.Parameter((init_weights_scale * torch.rand((n_layers, n_qubits, 3))))
        self.biases = nn.Parameter((init_weights_scale * torch.rand((n_layers, n_qubits, 3))))

    def get_angles(self, X):
        """
        multiplies X with weights and adds biases. Expands X such that for a single sample
        the features are encoded into the angles with repetition,
        e.g. for 2 qubits, 2 layers and 5 features:
        Rot(w1*x1+b1, w2*x2+b2, w3*x3+b3) Rot(w2*x2+b2, w3*x3+b3, w4*x4+b4)
        Rot(w4*x4+b4, w5*x5+b5, w1*x1+b1) Rot(w5*x5+b5, w1*x1+b1, w2*x2+b2)
        :param X: shape (n_samples, n_features)
        :return: angles of shape (n_samples, n_layers, n_qubits, 3)
        """
        n_samples = X.shape[0]
        n_features = X.shape[1]
        n_angles = self.n_qubits * self.n_layers * 3
        num_reps = n_angles // n_features + 1
        X_expanded = X.repeat(1, num_reps)
        X_expanded = X_expanded[:, :n_angles].reshape(n_samples, self.n_layers, self.n_qubits, 3).to(self.device)

        weights = self.weights
        bias = self.biases

        W = weights.unsqueeze(0).repeat(X.shape[0], 1, 1, 1)
        B = bias.unsqueeze(0).repeat(X.shape[0], 1, 1, 1)
        angles = X_expanded * W + B
        return angles

    def get_single_qubits_unitary(self, angles):
        """
        calculates states after applying a rot gate parametrized by [phi, theta, omega].
        Initial state is defined in init
        X shape = n_samples x n_qubits x 3 | (phi, theta, omega)"""
        n_qubits = angles.shape[1]
        n_samples = angles.shape[0]

        # https://docs.pennylane.ai/en/stable/code/api/pennylane.Rot.html
        # X =[phi, theta, omega] = RZ(omega)RY(theta)RZ(phi)
        ctheta = torch.cos(angles[:, :, 1] / 2)
        stheta = torch.sin(angles[:, :, 1] / 2)
        phi_plus_omega = ((angles[:, :, 0] + angles[:, :, 2]) / 2)
        phi_minus_omega = ((angles[:, :, 0] - angles[:, :, 2]) / 2)

        m00 = torch.exp(-1j * phi_plus_omega) * ctheta
        m01 = -torch.exp(1j * phi_minus_omega) * stheta
        m10 = torch.exp(-1j * phi_minus_omega) * stheta
        m11 = torch.exp(1j * phi_plus_omega) * ctheta
        M = torch.stack((m00, m01, m10, m11), dim=2).reshape(angles.shape[0], n_qubits, 2, 2)
        # kudos to SI
        M = self.get_system_unitary(M)
        return M

    def get_system_unitary(self, M):
        """
        :param M: tensor of single qubit unitaries of shape (n_samples, n_qubits, 2, 2)
        :return: M: tensor of system unitaries of shape (n_samples, 2**n_qubits, 2**n_qubits)
        """
        # kudos to SI
        n_samples = M.shape[0]
        n_qubits = self.n_qubits
        T = [chr(66 + n) if n < 25 else chr(72 + n) for n in range(51)]
        esum = ",".join([f"A{T[2 * n]}{T[2 * n + 1]}" for n in range(n_qubits)]) + \
               "->A" + \
               "".join([T[2 * n] for n in range(n_qubits)]) + \
               "".join([T[2 * n + 1] for n in range(n_qubits)])
        M = torch.einsum(esum, *(M[:, n, :, :] for n in range(n_qubits))).reshape(n_samples, 2 ** n_qubits,
                                                                                  2 ** n_qubits)
        return M

    def get_single_CNOT_matrix(self, control, target):
        """
        calculates states after applying two qubit gates
        gates: list of strings of the two qubit gates that will be applied
        qubit_map: list of tupels of qubit numbers (control, target) on which the gates will be applied, or string "ring", "next"
        """

        u = torch.zeros((2 ** self.n_qubits, 2 ** self.n_qubits))

        for i in range(2 ** self.n_qubits):
            bit_i = list(bin(i)[2:].zfill(self.n_qubits))
            if bit_i[control] == "1":
                bit_i[target] = str(int(bit_i[target]) ^ 1)
                bit_i_str = "".join(bit_i)
                index_1 = int(bit_i_str, 2)
                u[index_1, i] = 1
            else:
                u[i, i] = 1
        return u

    def get_Pauli_system_unitary(self, qubits, u_pauli, n_samples):
        identity = torch.eye(2).repeat(n_samples, 1, 1)
        partial_unitaries = []
        for i in range(self.n_qubits - 1, -1, -1):
            if i in qubits:
                partial_unitaries.append(u_pauli)
            else:
                partial_unitaries.append(identity)
        U = torch.stack(tuple(partial_unitaries), dim=1)
        M = self.get_system_unitary(U)
        return M

    def PauliX(self, qubits, n_samples):
        """
        gives unitary for Pauli X gate on qubits.
        qubits is a list of qubit numbers on which the gate will be applied
        """

        ux = torch.zeros((n_samples, 2, 2))
        ux[:, 0, 1] = 1
        ux[:, 1, 0] = 1
        M = self.get_Pauli_system_unitary(qubits, ux, n_samples)
        final_state = self.apply_unitary(M)
        return final_state

    def PauliY(self, qubits, n_samples):
        """
        gives unitary for Pauli Y gate on qubits.
        qubits is a list of qubit numbers on which the gate will be applied
        """

        uy = torch.zeros((self.n_samples, 2, 2), dtype=torch.complex64)
        uy[:, 0, 1] = 1j
        uy[:, 1, 0] = -1j
        M = self.get_Pauli_system_unitary(qubits, uy, n_samples)
        final_state = self.apply_unitary(M)
        return final_state

    def PauliZ(self, qubits, n_samples):
        """
        gives unitary for Pauli Y gate on qubits.
        qubits is a list of qubit numbers on which the gate will be applied
        """

        uz = torch.zeros((n_samples, 2, 2))
        uz[:, 0, 0] = 1
        uz[:, 1, 1] = -1
        M = self.get_Pauli_system_unitary(qubits, uz, n_samples)
        final_state = self.apply_unitary(M)
        return final_state

    def X(self, qubits, n_samples):
        return self.PauliX(qubits, n_samples)

    def Y(self, qubits, n_samples):
        return self.PauliY(qubits, n_samples)

    def Z(self, qubits, n_samples):
        return self.PauliZ(qubits, n_samples)

    def H(self, qubits, n_samples):
        """
        gives unitary for Hadamard gate on qubits.
        qubits is a list of qubit numbers on which the gate will be applied
        """

        uh = torch.ones((n_samples, 2, 2))
        uh[:, 1, 1] = -1
        uh = uh / np.sqrt(2)
        M = self.get_Pauli_system_unitary(qubits, uh, n_samples)
        final_state = self.apply_unitary(M)
        return final_state

    def Rot(self, X):
        """
        applies Rot gate on qubits.
        :param X: angles of shape n_samples x n_qubits x 3
        :return: final state after applying gate to all qubits
        """
        M = self.get_single_qubits_unitary(X)
        final_state = self.apply_unitary(M)
        return final_state

    def RX(self, Angles):
        """
        applies RX gate on qubits.
        RX(\theta) = RZ(-\pi/2) RY(\theta) RZ(\pi/2)
        from https://docs.pennylane.ai/en/stable/_modules/pennylane/ops/qubit/parametric_ops_single_qubit.html#RX
        :param Angles: angles of shape n_samples x n_qubits
        :return: final state after applying gate to all qubits
        """
        pis = torch.ones(Angles.shape[0], Angles.shape[1]) * np.pi
        X = torch.stack((pis / 2, Angles, -pis / 2), dim=2)
        M = self.get_single_qubits_unitary(X)
        final_state = self.apply_unitary(M)
        return final_state

    def RY(self, Angles):
        """
        applies RY gate on qubits.
        :param Angles: angles of shape n_samples x n_qubits
        :return: final state after applying gate to all qubits
        """
        zeros = torch.zeros(Angles.shape[0], Angles.shape[1])
        X = torch.stack((zeros, Angles, zeros), dim=2)
        M = self.get_single_qubits_unitary(X)
        final_state = self.apply_unitary(M)
        return final_state

    def RZ(self, Angles):
        """
        applies RZ gate on qubits.
        :param Angles: angles of shape n_samples x n_qubits
        :return: final state after applying gate to all qubits
        """
        zeros = torch.zeros(Angles.shape[0], Angles.shape[1])
        pis = torch.ones(Angles.shape[0], Angles.shape[1]) * np.pi
        X = torch.stack((Angles, pis, zeros), dim=2)
        M = self.get_single_qubits_unitary(X)
        final_state = self.apply_unitary(M)
        return final_state

    def CNOT(self, qubit_map, n_samples):
        """
        applies CNOT gate on qubits.
        :param qubit_map: list of tupels of qubit numbers (control, target) on which the gates will be applied, or string "ring", "next"
        :return: final state after applying gate to all qubits
        """
        if self.n_qubits < 2:
            return self.state
        else:
            qubit_map_string = ""
            if qubit_map == "next" or qubit_map == "ring":
                qubit_map_string = qubit_map
                qubit_map = [(i, i + 1) for i in range(self.n_qubits - 1)]
            if qubit_map_string == "ring" and self.n_qubits > 2:
                qubit_map.append((self.n_qubits - 1, 0))
            M = torch.eye(2 ** self.n_qubits)
            for control, target in qubit_map:
                M = self.get_single_CNOT_matrix(control, target).repeat(n_samples, 1, 1)
                final_state = self.apply_unitary(M)

            return final_state

    def SWAP(self, qubit_map):
        """
        applies SWAP gate on qubits.
        implemented as CNOT(0,1) CNOT(1,0) CNOT(0,1)
        :return: final state after applying gate to all qubits
        """
        if qubit_map == "next":
            qubit_map = [(i, i + 1) for i in range(self.n_qubits - 1)]
        if qubit_map == "ring":
            qubit_map = [(i, i + 1) for i in range(self.n_qubits1)]
            qubit_map.append((self.n_qubits1, 0))

        qubit_map_reversed = [(target, control) for control, target in qubit_map]
        M = self.get_CNOT_matrix(qubit_map)
        M_rev = self.get_CNOT_matrix(qubit_map_reversed)
        U = self.combine_layer_unitaries([M, M_rev, M])
        final_state = self.apply_unitary(U)
        return final_state

    def CZ(self, qubit_map, n_samples):
        if qubit_map == "next" or qubit_map == "ring":
            qubit_map = [(i, i + 1) for i in range(self.n_qubits - 1)]
        if qubit_map == "ring":
            qubit_map.append((self.n_qubits - 1, 0))

        u = torch.eye(self.n_qubits ** 2)
        for i in range(self.n_qubits):
            bit_i = bin(i)[2:].zfill(self.n_qubits)
            for control, target in qubit_map:
                if bit_i[-1 - control] == "1" and bit_i[-1 - target] == "1":
                    u[i, i] = -1
        U = u.repeat(n_samples, 1, 1)
        final_state = self.apply_unitary(U)
        return final_state

    def combine_layer_unitaries(self, unitaries):
        """
        :param unitaries: list of unitaries of shape (n_samples, 2**n_qubits, 2**n_qubits)
        :return: combined unitary of shape (n_samples, 2**n_qubits, 2**n_qubits)
        """
        unitary = torch.stack(unitaries).type(torch.complex64)
        n_layers = len(unitaries)
        # Perform batch matrix multiplication using einsum
        T = [chr(66 + n) if n < 25 else chr(72 + n) for n in range(51)]
        esum = ",".join([f"A{T[n]}{T[n + 1]}" for n in range(n_layers)]) + \
               "->A" + T[0] + T[n_layers]
        u_total = torch.einsum(esum, *(unitary[n, :, :, :] for n in range(n_layers))).type(torch.complex64)
        return u_total

    def apply_unitary(self, unitary):
        """
        applies unitary to self.state
        :param unitary: unitary of shape (n_samples, 2**n_qubits, 2**n_qubits)
        :return: final state after applying unitary
        """
        state = self.state.type(torch.complex128).to(self.device)
        unitary = unitary.type(torch.complex128).to(self.device)
        final_state = torch.einsum('ijk,ik->ij', unitary, state)
        self.state = final_state
        return final_state

    def expval_Z(self, wires=[0]):
        """
        measures expectation value of PauliZ gates on qubits given by wires
        :param wires: the measured wires
        :return: array of len(wires) expectation values of the given wires, in same order
        """
        state = self.state
        # print(state.shape)
        # state = torch.zeros_like(self.state)
        # state[:,1] = 1
        expval = torch.zeros((state.shape[0], len(wires)))
        amplitudes = torch.abs(state) ** 2
        sum_amp = torch.sum(amplitudes, axis=1)
        assert torch.allclose(sum_amp, torch.ones_like(sum_amp)), f'sum of amplitudes must equal 1, got {sum_amp}'
        len_state = state.shape[1]
        expval_idx = 0
        for w in wires:
            altern = 2 ** w
            fact = -1
            for idx in range(len_state):
                if (idx % altern) == 0:
                    fact *= -1
                expval[:, expval_idx] += fact * amplitudes[:, idx]
            expval_idx += 1
        return expval


    def __str__(self):
        s = f"--- StateVecSimTorch summary ---\n num_qubits: {self.n_qubits}\n num_layers: {self.n_layers}\n"
        return s


    def forward(self, X):
        self.state = torch.zeros(2 ** self.n_qubits)
        self.state[0] = 1.
        self.state = self.state.unsqueeze(1).repeat(1, X.shape[0]).T.type(torch.complex64)

        return self.state