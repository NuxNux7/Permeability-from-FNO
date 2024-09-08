 def train(self):

        self.model.train()

        # Placeholders used for capture
        static_input, static_target, static_mask, _ = next(iter(self.train_loader))

        # warmup
        s = torch.cuda.Stream()
        s.wait_stream(torch.cuda.current_stream())

        for i in range(3):
            static_input, static_target, static_mask = static_input.to(self.device), static_target.to(self.device), static_mask.to(self.device)

            self.optimizer.zero_grad()

            outputs = self.model(static_input)

            # Calculate loss
            loss = self.criterion(outputs, static_target, static_mask)
            loss.backward()
            nn.utils.clip_grad_value_(self.model.parameters(), clip_value=0.1)

            # Change weights
            self.optimizer.step()
            self.scheduler.step()
        
        torch.cuda.current_stream().wait_stream(s)

        # capture
        self.g = torch.cuda.CUDAGraph()
        self.optimizer.zero_grad(set_to_none=True)

        with torch.cuda.graph(self.g):
            static_input, static_target, static_mask = static_input.to(self.device), static_target.to(self.device), static_mask.to(self.device)

            static_outputs = self.model(static_input)

            # Calculate loss
            static_loss  = self.criterion(static_outputs, static_target, static_mask)
            static_loss .backward()
            nn.utils.clip_grad_value_(self.model.parameters(), clip_value=0.1)

            # Change weights
            self.optimizer.step()
            self.scheduler.step()

            total_loss += loss

        total_loss = 0
        for batch_idx, (inputs, targets, mask, _) in enumerate(self.train_loader):
            static_input.copy_(inputs)
            static_target.copy_(targets)
            static_mask.copy_(mask)

            self.g.replay()
            total_loss += static_loss.item()

        return total_loss / len(self.train_loader)