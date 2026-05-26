#!/usr/bin/env python3
"""
Dataset Statistics Visualization Script
Created: 2025-11-09 21:49

This script analyzes and visualizes the complete dataset statistics for the BEV
denoising and geolocation project, including:
- KITTI dataset (6 sequences, 4 weather conditions)
- NCLT dataset (12 dates, 4 weather conditions)
- Boreas dataset (4 sequences, 3 weather conditions)
"""

import os
import pickle
import numpy as np
import matplotlib
matplotlib.use('Agg')  # Non-interactive backend to avoid display errors
import matplotlib.pyplot as plt
from pathlib import Path
from collections import defaultdict

# Set matplotlib parameters for better visualization
plt.rcParams['font.size'] = 10
plt.rcParams['figure.dpi'] = 100
plt.rcParams['savefig.dpi'] = 300
plt.rcParams['savefig.bbox'] = 'tight'


class DatasetStatistics:
    """Analyze and visualize dataset statistics"""

    def __init__(self, base_dir):
        self.base_dir = Path(base_dir)
        self.data_dir = self.base_dir / 'data'
        self.stats = {
            'kitti': {},
            'nclt': {},
            'boreas': {}
        }

    def load_pickle_file(self, filepath):
        """Load pickle file safely"""
        try:
            with open(filepath, 'rb') as f:
                return pickle.load(f)
        except Exception as e:
            print(f"Warning: Could not load {filepath}: {e}")
            return None

    def analyze_kitti(self):
        """Analyze KITTI dataset statistics"""
        print("\n" + "="*80)
        print("ANALYZING KITTI DATASET")
        print("="*80)

        # Load denoising pairs
        pickle_path = self.data_dir / 'kitti_denoising_tuples.pkl'
        data = self.load_pickle_file(pickle_path)

        if data is None:
            print("KITTI data file not found!")
            return

        # Dataset information
        sequences = ['01-10-03-42', '02-10-03-14', '04-09-30-16',
                    '08-09-30-28', '09-09-30-33', '10-09-30-34']
        sequence_frames = {
            '01-10-03-42': 864,
            '02-10-03-14': 1687,
            '04-09-30-16': 136,
            '08-09-30-28': 1377,
            '09-09-30-33': 569,
            '10-09-30-34': 307
        }

        weather_conditions = ['orin', 'fog', 'rain', 'snow']

        # Calculate statistics
        total_frames = sum(sequence_frames.values())
        total_images = total_frames * len(weather_conditions)  # Each frame x 4 weather

        train_pairs = len(data.get('train', []))
        val_pairs = len(data.get('val', []))
        test_pairs = len(data.get('test', []))
        total_pairs = train_pairs + val_pairs + test_pairs

        # Weather distribution in pairs
        weather_dist = defaultdict(int)
        for split in ['train', 'val', 'test']:
            if split in data:
                for item in data[split]:
                    weather = item[2] if len(item) > 2 else 'unknown'
                    weather_dist[weather] += 1

        # Store statistics
        self.stats['kitti'] = {
            'sequences': sequences,
            'sequence_frames': sequence_frames,
            'total_frames': total_frames,
            'total_images': total_images,
            'weather_conditions': weather_conditions,
            'train_pairs': train_pairs,
            'val_pairs': val_pairs,
            'test_pairs': test_pairs,
            'total_pairs': total_pairs,
            'weather_distribution': dict(weather_dist)
        }

        # Print statistics
        print(f"\nSequences: {len(sequences)}")
        print(f"Total frames (clean): {total_frames:,}")
        print(f"Total images (all weather): {total_images:,}")
        print(f"\nSequence breakdown:")
        for seq, frames in sequence_frames.items():
            print(f"  {seq}: {frames:4d} frames")

        print(f"\nWeather conditions: {', '.join(weather_conditions)}")
        print(f"\nDenoising pairs:")
        print(f"  Training:   {train_pairs:6,} pairs")
        print(f"  Validation: {val_pairs:6,} pairs")
        print(f"  Test:       {test_pairs:6,} pairs")
        print(f"  Total:      {total_pairs:6,} pairs")

        print(f"\nWeather distribution in pairs:")
        for weather, count in sorted(weather_dist.items()):
            print(f"  {weather}: {count:6,} pairs ({count/total_pairs*100:.1f}%)")

    def analyze_nclt(self):
        """Analyze NCLT dataset statistics"""
        print("\n" + "="*80)
        print("ANALYZING NCLT DATASET")
        print("="*80)

        # Load denoising pairs
        pickle_path = self.data_dir / 'nclt_denoising_tuples.pkl'
        data = self.load_pickle_file(pickle_path)

        if data is None:
            print("NCLT data file not found!")
            return

        # Dataset information
        dates = ['2012-01-08', '2012-01-15', '2012-01-22', '2012-08-20',
                '2012-09-28', '2012-10-28', '2012-11-04', '2012-11-16',
                '2012-11-17', '2012-12-01', '2013-02-23', '2013-04-05']

        date_frames = {
            '2012-01-08': 1425,
            '2012-01-15': 1654,
            '2012-01-22': 750,
            '2012-08-20': 1123,
            '2012-09-28': 909,
            '2012-10-28': 783,
            '2012-11-04': 409,
            '2012-11-16': 998,
            '2012-11-17': 1148,
            '2012-12-01': 683,
            '2013-02-23': 958,
            '2013-04-05': 666
        }

        weather_conditions = ['orin', 'fog', 'rain', 'snow']

        # Calculate statistics
        total_frames = sum(date_frames.values())
        total_images = total_frames * len(weather_conditions)

        train_pairs = len(data.get('train', []))
        val_pairs = len(data.get('val', []))
        test_pairs = len(data.get('test', []))
        total_pairs = train_pairs + val_pairs + test_pairs

        # Weather distribution
        weather_dist = defaultdict(int)
        for split in ['train', 'val', 'test']:
            if split in data:
                for item in data[split]:
                    weather = item[2] if len(item) > 2 else 'unknown'
                    weather_dist[weather] += 1

        # Store statistics
        self.stats['nclt'] = {
            'dates': dates,
            'date_frames': date_frames,
            'total_frames': total_frames,
            'total_images': total_images,
            'weather_conditions': weather_conditions,
            'train_pairs': train_pairs,
            'val_pairs': val_pairs,
            'test_pairs': test_pairs,
            'total_pairs': total_pairs,
            'weather_distribution': dict(weather_dist)
        }

        # Print statistics
        print(f"\nDates: {len(dates)}")
        print(f"Total frames (clean): {total_frames:,}")
        print(f"Total images (all weather): {total_images:,}")
        print(f"\nDate breakdown:")
        for date, frames in date_frames.items():
            print(f"  {date}: {frames:4d} frames")

        print(f"\nWeather conditions: {', '.join(weather_conditions)}")
        print(f"\nDenoising pairs:")
        print(f"  Training:   {train_pairs:6,} pairs")
        print(f"  Validation: {val_pairs:6,} pairs")
        print(f"  Test:       {test_pairs:6,} pairs")
        print(f"  Total:      {total_pairs:6,} pairs")

        print(f"\nWeather distribution in pairs:")
        for weather, count in sorted(weather_dist.items()):
            print(f"  {weather}: {count:6,} pairs ({count/total_pairs*100:.1f}%)")

    def analyze_boreas(self):
        """Analyze Boreas dataset statistics"""
        print("\n" + "="*80)
        print("ANALYZING BOREAS DATASET")
        print("="*80)

        # Load denoising pairs
        pickle_path = self.data_dir / 'boreas_bev_denoising_pairs_spatial.pickle'
        data = self.load_pickle_file(pickle_path)

        if data is None:
            print("Boreas data file not found!")
            return

        # Dataset information (actual sequences)
        sequences = [
            'boreas-2020-12-01-13-26-snow',
            'boreas-2021-01-26-11-22-snow',
            'boreas-2021-04-08-12-44-clear',
            'boreas-2021-04-29-15-55-rain'
        ]

        sequence_frames = {
            'boreas-2020-12-01-13-26-snow': 2651,
            'boreas-2021-01-26-11-22-snow': 2645,
            'boreas-2021-04-08-12-44-clear': 2651,
            'boreas-2021-04-29-15-55-rain': 1620
        }

        weather_conditions = ['clear', 'snow', 'rain']

        # Calculate statistics
        total_frames = sum(sequence_frames.values())

        # Weather-based frame counts (real data, not simulated)
        weather_frames = {
            'clear': 2651,
            'snow': 5296,  # 2651 + 2645
            'rain': 1620
        }

        train_pairs = len(data.get('train', []))
        val_pairs = len(data.get('val', []))
        test_pairs = len(data.get('test', []))
        total_pairs = train_pairs + val_pairs + test_pairs

        # Weather distribution in pairs
        weather_dist = defaultdict(int)
        for split in ['train', 'val', 'test']:
            if split in data:
                for item in data[split]:
                    weather = item[2] if len(item) > 2 else 'unknown'
                    weather_dist[weather] += 1

        # Store statistics
        self.stats['boreas'] = {
            'sequences': sequences,
            'sequence_frames': sequence_frames,
            'total_frames': total_frames,
            'weather_conditions': weather_conditions,
            'weather_frames': weather_frames,
            'train_pairs': train_pairs,
            'val_pairs': val_pairs,
            'test_pairs': test_pairs,
            'total_pairs': total_pairs,
            'weather_distribution': dict(weather_dist)
        }

        # Print statistics
        print(f"\nSequences: {len(sequences)}")
        print(f"Total frames: {total_frames:,}")
        print(f"\nSequence breakdown:")
        for seq, frames in sequence_frames.items():
            weather = 'snow' if 'snow' in seq else ('rain' if 'rain' in seq else 'clear')
            print(f"  {seq}: {frames:4d} frames ({weather})")

        print(f"\nWeather distribution (frames):")
        for weather, count in sorted(weather_frames.items()):
            print(f"  {weather}: {count:6,} frames ({count/total_frames*100:.1f}%)")

        print(f"\nDenoising pairs:")
        print(f"  Training:   {train_pairs:6,} pairs")
        print(f"  Validation: {val_pairs:6,} pairs")
        print(f"  Test:       {test_pairs:6,} pairs")
        print(f"  Total:      {total_pairs:6,} pairs")

        print(f"\nWeather distribution in pairs:")
        for weather, count in sorted(weather_dist.items()):
            print(f"  {weather}: {count:6,} pairs ({count/total_pairs*100:.1f}%)")

    def print_summary(self):
        """Print overall summary statistics"""
        print("\n" + "="*80)
        print("OVERALL SUMMARY")
        print("="*80)

        total_frames = (
            self.stats['kitti'].get('total_frames', 0) +
            self.stats['nclt'].get('total_frames', 0) +
            self.stats['boreas'].get('total_frames', 0)
        )

        total_pairs = (
            self.stats['kitti'].get('total_pairs', 0) +
            self.stats['nclt'].get('total_pairs', 0) +
            self.stats['boreas'].get('total_pairs', 0)
        )

        print(f"\nTotal datasets: 3 (KITTI, NCLT, Boreas)")
        print(f"Total frames (clean): {total_frames:,}")
        print(f"Total denoising pairs: {total_pairs:,}")

        print(f"\nDataset distribution:")
        print(f"  KITTI:  {self.stats['kitti'].get('total_frames', 0):6,} frames "
              f"({self.stats['kitti'].get('total_frames', 0)/total_frames*100:.1f}%)")
        print(f"  NCLT:   {self.stats['nclt'].get('total_frames', 0):6,} frames "
              f"({self.stats['nclt'].get('total_frames', 0)/total_frames*100:.1f}%)")
        print(f"  Boreas: {self.stats['boreas'].get('total_frames', 0):6,} frames "
              f"({self.stats['boreas'].get('total_frames', 0)/total_frames*100:.1f}%)")

        print(f"\nPairs distribution:")
        print(f"  KITTI:  {self.stats['kitti'].get('total_pairs', 0):6,} pairs "
              f"({self.stats['kitti'].get('total_pairs', 0)/total_pairs*100:.1f}%)")
        print(f"  NCLT:   {self.stats['nclt'].get('total_pairs', 0):6,} pairs "
              f"({self.stats['nclt'].get('total_pairs', 0)/total_pairs*100:.1f}%)")
        print(f"  Boreas: {self.stats['boreas'].get('total_pairs', 0):6,} pairs "
              f"({self.stats['boreas'].get('total_pairs', 0)/total_pairs*100:.1f}%)")

    def visualize(self, output_path):
        """Create comprehensive visualization"""
        print("\n" + "="*80)
        print("GENERATING VISUALIZATIONS")
        print("="*80)

        # Create figure with subplots
        fig = plt.figure(figsize=(16, 12))

        # 1. Frame counts by dataset (bar chart)
        ax1 = plt.subplot(2, 3, 1)
        datasets = ['KITTI', 'NCLT', 'Boreas']
        frames = [
            self.stats['kitti'].get('total_frames', 0),
            self.stats['nclt'].get('total_frames', 0),
            self.stats['boreas'].get('total_frames', 0)
        ]
        colors = ['#1f77b4', '#ff7f0e', '#2ca02c']
        bars = ax1.bar(datasets, frames, color=colors, alpha=0.8, edgecolor='black')
        ax1.set_ylabel('Number of Frames', fontsize=11, fontweight='bold')
        ax1.set_title('Frame Counts by Dataset', fontsize=12, fontweight='bold')
        ax1.grid(axis='y', alpha=0.3, linestyle='--')

        # Add value labels on bars
        for bar in bars:
            height = bar.get_height()
            ax1.text(bar.get_x() + bar.get_width()/2., height,
                    f'{int(height):,}',
                    ha='center', va='bottom', fontsize=10, fontweight='bold')

        # 2. Dataset distribution (pie chart)
        ax2 = plt.subplot(2, 3, 2)
        ax2.pie(frames, labels=datasets, colors=colors, autopct='%1.1f%%',
                startangle=90, textprops={'fontsize': 11, 'fontweight': 'bold'})
        ax2.set_title('Dataset Distribution by Frames', fontsize=12, fontweight='bold')

        # 3. Weather distribution for each dataset
        ax3 = plt.subplot(2, 3, 3)

        # KITTI weather (in pairs)
        kitti_weather = self.stats['kitti'].get('weather_distribution', {})
        nclt_weather = self.stats['nclt'].get('weather_distribution', {})
        boreas_weather = self.stats['boreas'].get('weather_distribution', {})

        # Normalize to common weather names
        all_weathers = set()
        all_weathers.update(kitti_weather.keys())
        all_weathers.update(nclt_weather.keys())
        all_weathers.update(boreas_weather.keys())
        all_weathers = sorted(all_weathers)

        x = np.arange(len(all_weathers))
        width = 0.25

        kitti_vals = [kitti_weather.get(w, 0) for w in all_weathers]
        nclt_vals = [nclt_weather.get(w, 0) for w in all_weathers]
        boreas_vals = [boreas_weather.get(w, 0) for w in all_weathers]

        ax3.bar(x - width, kitti_vals, width, label='KITTI', color=colors[0], alpha=0.8, edgecolor='black')
        ax3.bar(x, nclt_vals, width, label='NCLT', color=colors[1], alpha=0.8, edgecolor='black')
        ax3.bar(x + width, boreas_vals, width, label='Boreas', color=colors[2], alpha=0.8, edgecolor='black')

        ax3.set_xlabel('Weather Condition', fontsize=11, fontweight='bold')
        ax3.set_ylabel('Number of Pairs', fontsize=11, fontweight='bold')
        ax3.set_title('Weather Distribution (Denoising Pairs)', fontsize=12, fontweight='bold')
        ax3.set_xticks(x)
        ax3.set_xticklabels(all_weathers, fontsize=10)
        ax3.legend(fontsize=10)
        ax3.grid(axis='y', alpha=0.3, linestyle='--')

        # 4. Train/Val/Test split by dataset
        ax4 = plt.subplot(2, 3, 4)

        train_vals = [
            self.stats['kitti'].get('train_pairs', 0),
            self.stats['nclt'].get('train_pairs', 0),
            self.stats['boreas'].get('train_pairs', 0)
        ]
        val_vals = [
            self.stats['kitti'].get('val_pairs', 0),
            self.stats['nclt'].get('val_pairs', 0),
            self.stats['boreas'].get('val_pairs', 0)
        ]
        test_vals = [
            self.stats['kitti'].get('test_pairs', 0),
            self.stats['nclt'].get('test_pairs', 0),
            self.stats['boreas'].get('test_pairs', 0)
        ]

        x = np.arange(len(datasets))
        width = 0.6

        p1 = ax4.bar(x, train_vals, width, label='Train', color='#3498db', alpha=0.9, edgecolor='black')
        p2 = ax4.bar(x, val_vals, width, bottom=train_vals, label='Val', color='#e74c3c', alpha=0.9, edgecolor='black')
        p3 = ax4.bar(x, test_vals, width, bottom=np.array(train_vals)+np.array(val_vals),
                    label='Test', color='#2ecc71', alpha=0.9, edgecolor='black')

        ax4.set_ylabel('Number of Pairs', fontsize=11, fontweight='bold')
        ax4.set_title('Train/Val/Test Split by Dataset', fontsize=12, fontweight='bold')
        ax4.set_xticks(x)
        ax4.set_xticklabels(datasets, fontsize=10)
        ax4.legend(fontsize=10)
        ax4.grid(axis='y', alpha=0.3, linestyle='--')

        # 5. Sequence/Date counts
        ax5 = plt.subplot(2, 3, 5)

        seq_counts = [
            len(self.stats['kitti'].get('sequences', [])),
            len(self.stats['nclt'].get('dates', [])),
            len(self.stats['boreas'].get('sequences', []))
        ]

        bars = ax5.bar(datasets, seq_counts, color=colors, alpha=0.8, edgecolor='black')
        ax5.set_ylabel('Number of Sequences/Dates', fontsize=11, fontweight='bold')
        ax5.set_title('Sequence/Date Counts', fontsize=12, fontweight='bold')
        ax5.grid(axis='y', alpha=0.3, linestyle='--')

        for bar in bars:
            height = bar.get_height()
            ax5.text(bar.get_x() + bar.get_width()/2., height,
                    f'{int(height)}',
                    ha='center', va='bottom', fontsize=11, fontweight='bold')

        # 6. Summary statistics table
        ax6 = plt.subplot(2, 3, 6)
        ax6.axis('off')

        table_data = [
            ['Metric', 'KITTI', 'NCLT', 'Boreas', 'Total'],
            ['Sequences/Dates',
             str(len(self.stats['kitti'].get('sequences', []))),
             str(len(self.stats['nclt'].get('dates', []))),
             str(len(self.stats['boreas'].get('sequences', []))),
             '-'],
            ['Clean Frames',
             f"{self.stats['kitti'].get('total_frames', 0):,}",
             f"{self.stats['nclt'].get('total_frames', 0):,}",
             f"{self.stats['boreas'].get('total_frames', 0):,}",
             f"{sum(frames):,}"],
            ['Weather Conditions',
             str(len(self.stats['kitti'].get('weather_conditions', []))),
             str(len(self.stats['nclt'].get('weather_conditions', []))),
             str(len(self.stats['boreas'].get('weather_conditions', []))),
             '-'],
            ['Train Pairs',
             f"{self.stats['kitti'].get('train_pairs', 0):,}",
             f"{self.stats['nclt'].get('train_pairs', 0):,}",
             f"{self.stats['boreas'].get('train_pairs', 0):,}",
             f"{sum(train_vals):,}"],
            ['Val Pairs',
             f"{self.stats['kitti'].get('val_pairs', 0):,}",
             f"{self.stats['nclt'].get('val_pairs', 0):,}",
             f"{self.stats['boreas'].get('val_pairs', 0):,}",
             f"{sum(val_vals):,}"],
            ['Test Pairs',
             f"{self.stats['kitti'].get('test_pairs', 0):,}",
             f"{self.stats['nclt'].get('test_pairs', 0):,}",
             f"{self.stats['boreas'].get('test_pairs', 0):,}",
             f"{sum(test_vals):,}"],
            ['Total Pairs',
             f"{self.stats['kitti'].get('total_pairs', 0):,}",
             f"{self.stats['nclt'].get('total_pairs', 0):,}",
             f"{self.stats['boreas'].get('total_pairs', 0):,}",
             f"{sum(train_vals) + sum(val_vals) + sum(test_vals):,}"]
        ]

        table = ax6.table(cellText=table_data, cellLoc='center', loc='center',
                         bbox=[0, 0, 1, 1])
        table.auto_set_font_size(False)
        table.set_fontsize(9)
        table.scale(1, 2)

        # Style header row
        for i in range(5):
            table[(0, i)].set_facecolor('#34495e')
            table[(0, i)].set_text_props(weight='bold', color='white')

        # Style data rows with alternating colors
        for i in range(1, len(table_data)):
            for j in range(5):
                if i % 2 == 0:
                    table[(i, j)].set_facecolor('#ecf0f1')
                else:
                    table[(i, j)].set_facecolor('#ffffff')

        ax6.set_title('Dataset Statistics Summary', fontsize=12, fontweight='bold', pad=20)

        # Overall title
        fig.suptitle('BEV Denoising and Geolocation Dataset Statistics',
                    fontsize=16, fontweight='bold', y=0.98)

        plt.tight_layout(rect=[0, 0, 1, 0.96])

        # Save figure
        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        plt.savefig(output_path, dpi=300, bbox_inches='tight')
        print(f"\nVisualization saved to: {output_path}")

        plt.close()


def main():
    """Main function"""
    base_dir = '/data/users/cxw/pro/clav'
    output_file = os.path.join(base_dir, 'docs', 'dataset_statistics.png')

    print("\n" + "="*80)
    print("DATASET STATISTICS AND VISUALIZATION TOOL")
    print("="*80)
    print(f"Base directory: {base_dir}")
    print(f"Output file: {output_file}")

    # Create analyzer
    analyzer = DatasetStatistics(base_dir)

    # Analyze each dataset
    analyzer.analyze_kitti()
    analyzer.analyze_nclt()
    analyzer.analyze_boreas()

    # Print summary
    analyzer.print_summary()

    # Generate visualization
    analyzer.visualize(output_file)

    print("\n" + "="*80)
    print("ANALYSIS COMPLETE")
    print("="*80)


if __name__ == '__main__':
    main()
