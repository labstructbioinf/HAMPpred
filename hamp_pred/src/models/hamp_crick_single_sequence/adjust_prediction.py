import numpy as np


class PredictionAdjust:
    def __init__(self, unknown=1000, max_helices=2, same_helices_length=True,
                 min_f1=0.8, max_unknown=0.49):
        self.unknown = unknown
        self.max_helices = max_helices
        self.same_helices_length = same_helices_length
        self.min_f1 = min_f1
        self.max_unknown = max_unknown

    def get_segments(self, seq, space_min=5, helix_min=7):
        spc = 0
        n = 0
        segments = []
        segment = []
        starts = []
        start_adding = False
        for n, pos in enumerate(seq):
            res = pos[0] if isinstance(pos, list) else pos
            if res == self.unknown:
                spc += 1
                if start_adding:
                    segment.append(pos)
            else:
                spc = 0
                start_adding = True
                segment.append(pos)
            if spc > space_min:
                if len(segment) > helix_min:
                    segments.append(segment)
                    starts.append(n)
                segment = []
                spc = 0
                start_adding = False
        if len(segment) > helix_min and segment not in segments:
            segments.append(segment)
            starts.append(n)
        segments, starts = self._adjust_segments((segments, starts))
        ranges = [(y - len(x) + 1, y + 1) for x, y in zip(segments, starts)]
        return segments, starts, ranges

    def segments_rot(self, left, right):
        rot = []
        for p, l in zip(left, right):
            if p != self.unknown and l != self.unknown:
                rot.append(p - l)
        return rot

    def _get_f1_and_mse(self, tr, pred):
        unk_tr, unk_pr = 0, 0
        cont = 0
        mse = 0
        for p, c in zip(tr, pred):
            if p != self.unknown:
                unk_tr += 1
            if c != self.unknown:
                unk_pr += 1
            if p != self.unknown and c != self.unknown:
                cont += 1
                mse += (p - c) ** 2
        if not cont:
            return 0, 0
        mse /= cont
        rec = cont / unk_tr if unk_tr else 1
        prec = cont / unk_pr if unk_pr else 1
        f1 = 2 / ((1 / rec) + (1 / prec))
        return f1, mse

    def mse_f1_depend(self, tr, pred):
        f1, mse = self._get_f1_and_mse(tr, pred)
        if f1 > self.min_f1:
            return mse
        return None

    def get_crick_tr_pred(self, data):
        pr = []
        tr = []
        pos_tr = []
        pos_pr = []
        for ind, row in data.iterrows():
            tpp, tps = [], []
            for t, p in zip(row['true'], row['prediction']):
                if t != self.unknown and p != self.unknown:
                    tpp.append(t)
                    tps.append(p)
            tr.append(np.mean(tpp))
            pr.append(np.mean(tps))
            pos_tr.extend(tpp)
            pos_pr.extend(tps)
        return tr, pr, pos_tr, pos_pr

    def get_rot_tr_pred(self, data):
        pr = []
        tr = []
        for ind, row in data.iterrows():
            tr_segments, tr_pos, ranges = self.get_segments(row['true'])
            pred_segments, pred_pos, ranges = self.get_segments(row['prediction'])
            tr_rot, pred_rot = self.segments_rot(*tr_segments), self.segments_rot(*pred_segments)
            tr.append(np.mean(tr_rot))
            pr.append(np.mean(pred_rot))
        return tr, pr

    def flatten_prediction(self, data):
        data['prediction'] = data['prediction'].apply(lambda x: np.reshape(x, len(x)))
        return data

    def get_with_f1_depend(self, res):
        res['mse_f1'] = res.apply(lambda x: self.mse_f1_depend(x['true'], x['prediction']), axis=1)
        correct = res[res['mse_f1'].notnull()]
        return correct

    def _adjust_segments(self, data):
        segments, ends = data
        for n, seg in enumerate(segments):
            i = 0
            while seg[i] == self.unknown:
                i += 1
            j = -1
            while seg[j] == self.unknown:
                j -= 1
            ends[n] = ends[n] + j + 1
            if j == -1:
                segments[n] = seg[i:]
            else:
                segments[n] = seg[i:j + 1]

        new_segments, new_data = [], []
        for n, seg in enumerate(segments):
            if not seg.count(self.unknown) > len(seg) * self.max_unknown:
                new_segments.append(seg)
                new_data.append(ends[n])
        if not new_segments:
            return new_segments, new_data
        org = [(n, seq) for n, seq in enumerate(segments)]
        org.sort(key=len, reverse=True)
        new_data = [new_data[pos[0]] for pos in org]
        org, new_data = org[:self.max_helices], new_data[:self.max_helices]
        if self.same_helices_length:
            adj = self._align_sequences(org)
            for pos, (left, right) in adj.items():
                new_segments[pos] = [self.unknown] * left + new_segments[pos] + [self.unknown] * right
                new_data[pos] = new_data[pos] + right
        return new_segments, new_data

    def _align_sequences(self, org):
        adjusted = {org[0][0]: (0, 0)}
        for i in range(1, len(org)):
            seq = org[i][1]
            pos = org[i][0]
            max_seq = org[0][1]
            best_f1 = 10e10
            left, right = 0, 0
            for j in range(len(max_seq) - len(seq)):
                f1, mse = self._get_f1_and_mse(max_seq[j:len(seq)], seq)
                if f1 < best_f1:
                    left = j
                    right = len(max_seq) - len(seq) - j
                    best_f1 = f1
            adjusted[pos] = (left, right)
        return adjusted

    def get_rotation(self, left=None, right=None):
        if not left or not right:
            return None
        f1, mse = self._get_f1_and_mse(left, right)
        if f1 > self.min_f1:
            return [x-y for x, y in zip(left, right) if x != self.unknown and y!= self.unknown]
        return None