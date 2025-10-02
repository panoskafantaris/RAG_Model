import { Component } from '@angular/core';
import { CommonModule } from '@angular/common';
import { ChatService } from '../../services/chat.service';
import { ChatSession } from '../../models/chat';
import { ReactiveFormsModule, FormControl } from '@angular/forms';
import { ButtonModule } from 'primeng/button';
import { InputTextModule } from 'primeng/inputtext';

@Component({
  selector: 'app-chat-window',
  standalone: true,
  imports: [CommonModule, ReactiveFormsModule, ButtonModule, InputTextModule],
  templateUrl: './chat-window.html',
  styleUrl: './chat-window.scss'
})
export class ChatWindowComponent {
  session?: ChatSession;
  message = new FormControl('');

  constructor(private chatService: ChatService) {
    this.chatService.getActiveSession().subscribe(s => this.session = s);
  }

  sendMessage() {
    if (this.session && this.message.value?.trim()) {
      this.chatService.sendMessage(this.session.id, this.message.value);
      this.message.reset();
    }
  }
}
