#!/usr/bin/env python
"""
Insert the compression hook into OmniVinci's modeling_vila.py.

Where and why:
    _embed() builds the LLM input by walking the video segment by segment. Inside that
    loop it slices out the video tokens and the audio tokens belonging to segment j:

        _vis_fea = media_embeds["video"][idx][vis_end:_vis_fea_end]
        _aud_fea = media_embeds["sound"][idx][aud_end:_aud_fea_end]

    That is exactly the time window OmniZip operates on, already split by modality.
    The hook is inserted between the slicing and the concatenation.

Why not prune earlier, at the encoder output:
    The slice boundaries are derived from the unpruned tensor:

        vis_fea_len_per_frame = media_embeds["video"][idx].shape[0] / expected_frame_count

    Removing tokens before this point makes the per-frame length wrong and every segment
    boundary shifts. Pruning after the slice leaves that arithmetic untouched.

The edit is a single exact-match block replacement, so it refuses to apply if the model
file differs from the version this was written against. Idempotent, and revertible from
the .orig backup.

Usage:
    python patch_model.py            # apply
    python patch_model.py --check    # report status only
    python patch_model.py --revert   # restore the original
"""
import argparse
import os
import shutil
import sys
from pathlib import Path

MARKER = "# --- OMNIZIP HOOK ---"

OLD = '''                    for j in range(len(segment_vis_indices_list)):
                        _vis_aud_fea = []
                        if len(segment_vis_indices_list[j]) > 0:
                            _new_frames = [int(np.ceil((_frame+1) * vis_fea_len_per_frame)) for _frame in segment_vis_indices_list[j]]
                            _vis_fea_end = _new_frames[-1]
                            # Ensure we don't exceed the available features
                            _vis_fea_end = min(_vis_fea_end, media_embeds["video"][video_embeds_idx].shape[0])
                            if j == len(segment_vis_indices_list) - 1 and i == len(video_info) - 1 and k == len(video_info[i]) - 1 and not _vis_fea_end == media_embeds["video"][video_embeds_idx].shape[0]: 
                                print(f"Warning: The number of last interleaved video features does not match the video feature length. Expected: {media_embeds['video'][video_embeds_idx].shape[0]}, Got: {_vis_fea_end}")
                                _vis_fea_end = media_embeds["video"][video_embeds_idx].shape[0]
                            _vis_fea = media_embeds["video"][video_embeds_idx][vis_end:_vis_fea_end]
                            vis_end = _vis_fea_end
                            _vis_aud_fea.append(_vis_fea)
                        _vis_aud_fea.append(sep_embed)
                        if len(segment_aud_indices_list[j]) > 0:
                            _new_audio_indices = [int(np.ceil(_fea * aud_fea_len_per_stft_frame)) for _fea in segment_aud_indices_list[j]]
                            _aud_fea_end = _new_audio_indices[-1]
                            # Ensure we don't exceed the available features
                            _aud_fea_end = min(_aud_fea_end, media_embeds["sound"][video_sound_embeds_idx].shape[0])
                            _aud_fea = media_embeds["sound"][video_sound_embeds_idx][aud_end:_aud_fea_end]
                            _vis_aud_fea.append(_aud_fea)
                            aud_end = _aud_fea_end
                        _vis_aud_fea.append(sep_embed)
                        _new_video_embed.append(torch.cat(_vis_aud_fea, dim=0))'''

NEW = '''                    for j in range(len(segment_vis_indices_list)):
                        _vis_aud_fea = []
                        _vis_fea = None
                        _aud_fea = None
                        if len(segment_vis_indices_list[j]) > 0:
                            _new_frames = [int(np.ceil((_frame+1) * vis_fea_len_per_frame)) for _frame in segment_vis_indices_list[j]]
                            _vis_fea_end = _new_frames[-1]
                            # Ensure we don't exceed the available features
                            _vis_fea_end = min(_vis_fea_end, media_embeds["video"][video_embeds_idx].shape[0])
                            if j == len(segment_vis_indices_list) - 1 and i == len(video_info) - 1 and k == len(video_info[i]) - 1 and not _vis_fea_end == media_embeds["video"][video_embeds_idx].shape[0]: 
                                print(f"Warning: The number of last interleaved video features does not match the video feature length. Expected: {media_embeds['video'][video_embeds_idx].shape[0]}, Got: {_vis_fea_end}")
                                _vis_fea_end = media_embeds["video"][video_embeds_idx].shape[0]
                            _vis_fea = media_embeds["video"][video_embeds_idx][vis_end:_vis_fea_end]
                            vis_end = _vis_fea_end
                        if len(segment_aud_indices_list[j]) > 0:
                            _new_audio_indices = [int(np.ceil(_fea * aud_fea_len_per_stft_frame)) for _fea in segment_aud_indices_list[j]]
                            _aud_fea_end = _new_audio_indices[-1]
                            # Ensure we don't exceed the available features
                            _aud_fea_end = min(_aud_fea_end, media_embeds["sound"][video_sound_embeds_idx].shape[0])
                            _aud_fea = media_embeds["sound"][video_sound_embeds_idx][aud_end:_aud_fea_end]
                            aud_end = _aud_fea_end
                        # --- OMNIZIP HOOK ---
                        # Both modalities for this time window are now sliced out. Compress
                        # here: the slice boundaries above were computed from the unpruned
                        # tensors, so they stay correct. A no-op unless a method is active.
                        try:
                            from omnizip_ov import compress_segment as _oz_compress
                            _vis_fea, _aud_fea = _oz_compress(_vis_fea, _aud_fea,
                                                              segment_index=j,
                                                              num_segments=len(segment_vis_indices_list))
                        except ImportError:
                            pass
                        # --- END OMNIZIP HOOK ---
                        if _vis_fea is not None:
                            _vis_aud_fea.append(_vis_fea)
                        _vis_aud_fea.append(sep_embed)
                        if _aud_fea is not None:
                            _vis_aud_fea.append(_aud_fea)
                        _vis_aud_fea.append(sep_embed)
                        _new_video_embed.append(torch.cat(_vis_aud_fea, dim=0))'''


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model-path", default=os.environ.get(
        "OV_MODEL_DIR", "/mnt/hdd2/trung/projects/new_users/Nghia/omnivinci"))
    ap.add_argument("--check", action="store_true")
    ap.add_argument("--revert", action="store_true")
    args = ap.parse_args()

    target = Path(args.model_path) / "modeling_vila.py"
    backup = target.with_suffix(".py.orig")

    if not target.exists():
        print("ERROR: %s not found" % target)
        sys.exit(1)

    src = target.read_text(encoding="utf-8")
    patched = MARKER in src

    if args.check:
        print("file    : %s" % target)
        print("patched : %s" % ("yes" if patched else "no"))
        print("backup  : %s" % ("present" if backup.exists() else "missing"))
        sys.exit(0 if patched else 1)

    if args.revert:
        if not backup.exists():
            print("ERROR: no backup at %s" % backup)
            sys.exit(1)
        shutil.copy2(backup, target)
        print("Reverted %s from backup." % target)
        return

    if patched:
        print("Already patched, nothing to do.")
        return

    if OLD not in src:
        print("ERROR: the expected code block was not found in %s." % target)
        print("The model file differs from the version this patch was written against.")
        print("Compare the segment loop in _embed() before editing anything by hand.")
        sys.exit(1)

    if not backup.exists():
        shutil.copy2(target, backup)
        print("Backup written to %s" % backup)

    target.write_text(src.replace(OLD, NEW), encoding="utf-8")
    print("Patched %s" % target)
    print("Transformers re-copies this file into its module cache on the next load,")
    print("so no further action is needed.")


if __name__ == "__main__":
    main()
