import { Injectable } from '@angular/core';

export interface LyricsLine {
  text: string;
  time: number;
  duration?: number;
}

export interface ChordData {
  time: number;
  chord: string;
}

export interface BarData {
  bar_index: number;
  time: number;
  chords: string[];
}

@Injectable({
  providedIn: 'root'
})
export class ApiService {
  private baseUrl = 'http://127.0.0.1:8000';

  constructor() {}

  async selectFile(): Promise<any> {
    const res = await fetch(`${this.baseUrl}/api/select-file`);
    return res.json();
  }

  async extractChords(path: string): Promise<any> {
    const res = await fetch(`${this.baseUrl}/api/extract-chords`, {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({ path })
    });
    return res.json();
  }

  async extractYoutube(url: string): Promise<any> {
    const res = await fetch(`${this.baseUrl}/api/extract-youtube`, {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({ url })
    });
    return res.json();
  }

  async getExtractionStatus(taskId: string): Promise<any> {
    const res = await fetch(`${this.baseUrl}/api/extract-chords/status/${taskId}`);
    return res.json();
  }

  async generateChordsheet(chords: ChordData[], lyrics: LyricsLine[], duration: number): Promise<any> {
    const res = await fetch(`${this.baseUrl}/api/generate-chordsheet`, {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({ chords, lyrics, duration })
    });
    return res.json();
  }
}
